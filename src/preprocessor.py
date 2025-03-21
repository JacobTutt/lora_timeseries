from transformers import AutoTokenizer
import numpy as np
import h5py
import logging 
logging.basicConfig(level=logging.INFO,  format="%(levelname)s - %(message)s",  datefmt="%H:%M:%S")

def preprocessor(prey_preditor_path, percentile= 95, decimal_places=2, split_fraction =0.8 , shuffle = False, tokeniser_model = None):
    """
    Preprocesses predator-prey time series data from an HDF5 file by:
    1. Loading the data and verifying required fields.
    2. Scaling all values based on the 95th percentile (default) to ensure data fits within a standard range.
    3. Encoding the data into a string format that can be tokenised by QWEN2.5-0.5B-Instruct.
    Parameters:
    ----------
    prey_preditor_path : str
        Path to the HDF5 file containing the Lotka-Volterra simulated data.
    percentile : int, optional (default=95)
        The percentile value used for determining the scaling factor (alpha).
        Ensures that `percentile%` of the data falls within the standard range.
    decimal_places : int, optional (default=2)
        Number of decimal places to round the scaled data to.

    Returns:
    -------
    str
        Encoded multi-variate time series data in the format:
        "prey1,predator1; prey2,predator2; ...".

    Raises:
    ------
    RuntimeError
        If the HDF5 file cannot be opened or required datasets are missing.
    ValueError
        If the dimensions of prey, predator, and time data do not match.
    """
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
        np.random.shuffle(encoded_data)

    # Split the data into training and validation sets. 
    split_index = int(len(encoded_data) * split_fraction)
    train_data = encoded_data[:split_index]
    val_data = encoded_data[split_index:]



    ### This next part is purely for verification, and it does not return tokenised data

    # This is stored in encoded data but dynamically generated for the display
    numerical_string = ';'.join(encoded_data[0].split(';')[:5])

    # Now import the tokeniser from the Qwen model if it is provided for display purposes
    if tokeniser_model is not None:
        # This is simply to show the encoding of the data, it is not required for the function
        encoded_data = tokeniser_model(numerical_string, return_tensors="pt")["input_ids"].tolist()[0]

        # Print an example of the data encoding
        logging.info(f"An example of the data encoding is shown below:\n"
                    f"Before Encoding - First 5 Prey Data: {[f'{prey:.{decimal_places}f}' for prey in scaled_trajectories[0, :5, 0]]}\n"
                    f"Before Encoding - First 5 Predator Data: {[f'{predator:.{decimal_places}f}' for predator in scaled_trajectories[0, :5, 1]]}\n"
                    f"After Encoding to String - First 5 Entries: {numerical_string}\n"
                    f"After Encoding to Tokenised: {encoded_data}")

        
    else: 
        # Print an example of the data encoding
        logging.info(f"An example of the data encoding is shown below:\n"
                    f"Before Encoding - First 5 Prey Data: {[f'{prey:.{decimal_places}f}' for prey in scaled_trajectories[0, :5, 0]]}\n"
                    f"Before Encoding - First 5 Predator Data: {[f'{predator:.{decimal_places}f}' for predator in scaled_trajectories[0, :5, 1]]}\n"
                    f"After Encoding to String - First 5 Entries: {numerical_string}")

    return train_data, val_data










def decoder(encoded_string):
    """
    Decodes a multivariate time series string into two NumPy arrays:
    - Prey population data
    - Predator population data

    This function reverses the encoding process, extracting the prey and predator 
    population values from a structured string representation.

    Parameters
    ----------
    encoded_string : str
        A semicolon-separated string of prey-predator pairs in the format:
        `"prey_1,predator_1; prey_2,predator_2; ..."`.
        Each pair represents the population values of prey and predator at a given time step.

    Returns
    -------
    tuQWEN le[np.ndarray, np.ndarray]
        - prey_array : np.ndarray
            A NumPy array containing the prey population values as floats.
        - predator_array : np.ndarray
            A NumPy array containing the predator population values as floats.

    Raises
    ------
    ValueError
        If an incorrectly formatted input string is provided.
    """

    # Split the string into individual "prey,predator" pairs
    data_pairs = encoded_string.split(";")

    # Extract prey and predator values separately
    prey_data = []
    predator_data = []

    for pair in data_pairs:
        if pair.strip():  # Ensure the pair is not empty (to handle trailing semicolons)
            # if there is an odd number at the end. 
            if "," not in pair:
                continue
            try:
                prey, predator = pair.split(",")  
                # Convert to float for numerical use (remove any spaces)
                prey_data.append(float(prey.strip()))  
                predator_data.append(float(predator.strip()))
            except ValueError:
                raise ValueError(f"Invalid data format encountered in: '{pair}'. Ensure correct structure.")

    # Convert lists to NumPy arrays
    prey_array = np.array(prey_data, dtype=np.float32)
    predator_array = np.array(predator_data, dtype=np.float32)

    return prey_array, predator_array




