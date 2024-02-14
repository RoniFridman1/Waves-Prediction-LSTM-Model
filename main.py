from LSTMDataManager import LSTMDataManager
import sys
from utils import update_latest_data_from_db
import os
from dotenv import load_dotenv
import pathlib
load_dotenv()

if __name__ == "__main__":
    py_file_path, function, location, target_variable = sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3]
    base_location_folder = str(pathlib.Path(sys.argv[0]).parent)
    os.makedirs(os.path.join(base_location_folder, "text_outputs"), exist_ok=True)
    full_data_path = os.path.join(base_location_folder, 'datasets', 'cameri_buoy_data', f'{location}.csv')
    wind_data_path = os.path.join(base_location_folder, 'datasets', 'cameri_buoy_data', f'{location}_wind.csv')
    lstm_data_manager = LSTMDataManager(location, target_variable, base_location_folder)


    function_blocks = function.split("_")
    if function == 'build':
        lstm_data_manager.build_new_model()
    elif function == "update":
        update_latest_data_from_db(full_data_path)
    elif function == 'predict':
        lstm_data_manager.predict_latest_available_data()
    elif function == 'build_predict_upload':
        lstm_data_manager.build_new_model()
        update_latest_data_from_db(full_data_path)
        lstm_data_manager.predict_latest_available_data()
