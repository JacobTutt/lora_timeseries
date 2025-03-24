from transformers import AutoTokenizer
import numpy as np
import h5py
import logging 
logging.basicConfig(level=logging.INFO,  format="%(levelname)s - %(message)s",  datefmt="%H:%M:%S")

def preprocessor(prey_preditor_path, percentile= 90, decimal_places=3, train_fraction =0.7, validation_fraction= 0.15 , shuffle = False, tokeniser_model = None, print_summary = True):
    """
    Preprocesses Lotka-Volterra predator-prey time series data from an HDF5 file.

    This function performs the following steps:
    1. Loads trajectory and time data from an HDF5 file.
    2. Scales the prey and predator populations using a percentile-based normalization.
    3. Converts the scaled data into a string-based sequence format suitable for tokenisation by Qwen-like models.
    4. Optionally shuffles the dataset.
    5. Splits the data into train, validation, and test sets.
    6. Optionally prints a summary including tokenized examples if a tokenizer is provided.

    Parameters
    ----------
    prey_preditor_path : str
        Path to the HDF5 file containing the simulated time series data.
        Expected to include datasets named "trajectories" and "time".

    percentile : int, optional (default=90)
        Percentile used to compute a scaling factor. Ensures `percentile`% of values fall in a compressed range.

    decimal_places : int, optional (default=3)
        Number of decimal places to round the scaled values before encoding to string.

    train_fraction : float, optional (default=0.8)
        Fraction of the dataset to use for training. Must be less than 1 - `validation_fraction`.

    validation_fraction : float, optional (default=0.1)
        Fraction of the dataset to use for validation. The remainder is used for testing.

    shuffle : bool, optional (default=False)
        Whether to randomly shuffle the trajectories before splitting.

    tokeniser_model : transformers.PreTrainedTokenizer or None, optional
        Tokenizer object (e.g., from Hugging Face) used for printing a preview of how the encoded string would be tokenized.
        This is only used for logging/debugging; tokenization is not returned.

    print_summary : bool, optional (default=True)
        If True, logs a summary of the preprocessing steps, including an example of the scaled and tokenized output.

    Returns
    -------
    train_data : list of str
        List of encoded strings for training. Each string represents one trajectory in the format:
        `"prey1,predator1;prey2,predator2;..."`

    val_data : list of str
        List of encoded strings for validation.

    test_data : list of str
        List of encoded strings for testing.

    Raises
    ------
    RuntimeError
        If the HDF5 file cannot be opened or required datasets are missing.

    ValueError
        If:
        - The scaling factor is non-positive.
        - The sum of `train_fraction` and `validation_fraction` exceeds or equals 1.
    """
    if train_fraction + validation_fraction >= 1:
        raise ValueError("The sum of the training and validation fractions must be less than 1 the remaining fraction is used for the test set")

    # Attempt to open the file and import the trajectories and time points
    try:
        with h5py.File(prey_preditor_path, 'r') as full_data:

            # Ensure that the HDF5 file contains the necessary data - time and trajectories
            if "trajectories" not in full_data or "time" not in full_data:
                raise KeyError("The inputed file path does not contain required data: 'trajectories' and 'time'.")
            
            # Extract the data
            trajectories = full_data["trajectories"][:] # Shape: (num_trajectories, num_time_points, 2)
            time_values = full_data["time"][:]          # Shape: (num_time_points,)

    except (OSError, KeyError) as e:
        raise RuntimeError(f"Failed to load datafile: {e}")
    
    # Report the shape of the data 
    logging.info(f"File loaded successfully. Trajectories shape: {trajectories.shape}, Time points shape: {time_values.shape}")

    # Apply scaling to the data
    # Allows us to not worry about wasting tokens for large inputs and also allows us to use the same model for different scales of data
    def scaling_data(data, percentile): 
        """
        Scales the predator-prey population data using the specified percentile.

        This ensures that a give percentile (95% default) of values fall within the model's expected range,
        reducing variance across different scales of data.

        Parameters:
        ----------
        data : np.ndarray
            The original trajectories of prey and predator populations.
            Shape: (num_trajectories, num_time_points, 2).
        percentile : int
            The percentile value used to determine the scaling factor.

        Returns:
        -------
        tuple:
            - scaled_data (np.ndarray): The scaled data.
            - alpha (float): The computed scaling factor used for normalisation.

        Raises:
        ------
        ValueError:
            If the computed scaling factor is zero or invalid.
        """ 

        # Compute the max 95 percentile values for both and pick the maximium
        # From the distributions and common sense with be the predator but both preformed and calulated for robustness
        # Flatten both prey and predator populations into a single array
        merged_data = np.concatenate([data[:, :, 0].flatten(), data[:, :, 1].flatten()])

        # Compute the percentile across both
        percentile_scale = np.percentile(merged_data, percentile)

        # Set the scaling factor (alpha)
        alpha = percentile_scale / 10

        if alpha <= 0:
            raise ValueError("Computed scaling factor (alpha) is non-positive. Check input data.")

        logging.info(f"Scaling data by alpha={alpha}, ensuring {percentile}% of values fit within the model's expected range.")


        # Scale the data
        scaled_data = data/ alpha

        logging.info(f"Data scaled to {decimal_places} decimal places")

        return scaled_data, alpha
    
    # Preform scaling 
    scaled_trajectories, alpha = scaling_data(trajectories, percentile)


    # Translate our array formated time series data into a string format
    def multivar_timeseries_encode(scaled_data_input, decimal_places):
        """
        Encodes the multivariate time series data into a **list of strings**, where each string
        represents one trajectory.

        The format for each string is:
        `"prey_1,predator_1; prey_2,predator_2; ..."`

        Parameters:
        ----------
        scaled_data_input : np.ndarray
            The scaled time series data after normalization.
            Shape: (num_trajectories, num_time_points, 2).
        decimal_places : int
            The number of decimal places to format the output.

        Returns:
        -------
        list of str
            A list where each item represents one trajectory.
        """

        encoded_list = []
        # Iterate over trajectories
        for i in range(scaled_data_input.shape[0]): 
            # Prey and predator populations for trajectory i
            prey_input = scaled_data_input[i, :, 0] 
            predator_input = scaled_data_input[i, :, 1] 

            # Explicitly format each number to `decimal_places` decimal places
            trajectory_string = ";".join(
                f"{prey_t:.{decimal_places}f},{predator_t:.{decimal_places}f}"
                for prey_t, predator_t in zip(prey_input, predator_input)
            )
            encoded_list.append(trajectory_string)

        return encoded_list

    # Execute encoding of data
    encoded_data = multivar_timeseries_encode(scaled_trajectories, decimal_places)


    # Preform a random shuffling
    if shuffle:
        logging.info("Shuffling the data so that the order of the data does not affect the model")
        np.random.shuffle(encoded_data)

    # Split the data into training, validation and test sets. 
    logging.info(f"Splitting the data into training, validation, and test sets with fractions: {train_fraction}, {validation_fraction}, {1 - train_fraction - validation_fraction}")
    train_index = int(len(encoded_data) * train_fraction)
    val_index = int(len(encoded_data) * (train_fraction + validation_fraction))


    train_data = encoded_data[:train_index]
    val_data = encoded_data[train_index:val_index]
    test_data = encoded_data[val_index:]


    ### This next part is purely for verification, and it does not return tokenised data
    if print_summary:
        # This is stored in encoded data but dynamically generated for the display
        numerical_string = ';'.join(encoded_data[0].split(';')[:5])

        # Now import the tokeniser from the Qwen model if it is provided for display purposes
        if tokeniser_model is not None:
            # This is simply to show the encoding of the data, it is not required for the function
            encoded_data = tokeniser_model(numerical_string, return_tensors="pt")["input_ids"].tolist()[0]

            # Print an example of the data encoding
            log_message = (f"An example of the data encoding is shown below:\n"
                        f"Before Encoding - First 5 Prey Data: {[f'{prey:.{decimal_places}f}' for prey in scaled_trajectories[0, :5, 0]]}\n"
                        f"Before Encoding - First 5 Predator Data: {[f'{predator:.{decimal_places}f}' for predator in scaled_trajectories[0, :5, 1]]}\n")
            if not shuffle:
                log_message += f"After Encoding to String - First 5 Entries: {numerical_string}\n"
            log_message += f"After Encoding to Tokenised: {encoded_data}"
            logging.info(log_message)

            
        else: 
            # Print an example of the data encoding
            log_message = (f"An example of the data encoding is shown below:\n"
                        f"Before Encoding - First 5 Prey Data: {[f'{prey:.{decimal_places}f}' for prey in scaled_trajectories[0, :5, 0]]}\n"
                        f"Before Encoding - First 5 Predator Data: {[f'{predator:.{decimal_places}f}' for predator in scaled_trajectories[0, :5, 1]]}\n")
            if not shuffle:
                log_message += f"After Encoding to String - First 5 Entries: {numerical_string}"
            logging.info(log_message)

    return train_data, val_data, test_data



