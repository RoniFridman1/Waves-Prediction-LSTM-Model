import copy
import datetime
import os.path
import sys
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import utils
from SequenceDataset import SequenceDataset
from torch.utils.data import DataLoader
import tqdm
from utils import *
from matplotlib import pyplot as plt

torch.manual_seed(int(os.getenv("TORCH_SEED")))


class LSTMDataManager:
    def __init__(self, location, target_variable, base_location_folder):
        load_dotenv(override=True)
        # forward time to predict
        self.target_variable = target_variable
        self.forecast_lead_hours = int(os.getenv("FORECAST_LEAD_HOURS"))
        self.location = location

        # paths for running on db77
        self.full_data_path = os.path.join(base_location_folder,"datasets",'cameri_buoy_data',f"{location}.csv")
        self.wind_data_path = os.path.join(base_location_folder,"datasets",'cameri_buoy_data',f"{location}_wind.csv")
        self.swell_data_path = os.path.join(base_location_folder,"datasets",'cameri_buoy_data',f"{location}_swell.csv")
        self.ims_data_path = os.path.join(base_location_folder,"datasets",'cameri_buoy_data',f"{location}_ims.csv")
        self.output_path = base_location_folder + f"/outputs/{location}_{self.forecast_lead_hours}h_{target_variable}_forecast"
        self.images_output_path = os.path.join(self.output_path, "images")
        self.csv_output_path = os.path.join(self.output_path, "csv files")
        self.new_data_training_path = f"{self.csv_output_path}/{location}_data_for_{target_variable}_forecast_{self.forecast_lead_hours}h.csv"
        self.forecast_groundtruth = f"{self.csv_output_path}/{location}_{target_variable}_groundtruth_{self.forecast_lead_hours}h.csv"
        self.model_path = f"{self.output_path}/{location}_shallow_reg_lstm_{target_variable}_{self.forecast_lead_hours}h.pth"
        self.train_loader_path = f"{self.output_path}/{location}_train_loader_{self.forecast_lead_hours}h_forecast_{target_variable}.pth"
        self.test_loader_path = f"{self.output_path}/{location}_test_loader_{self.forecast_lead_hours}h_forecast_{target_variable}.pth"
        self.train_eval_loader_path = f"{self.output_path}/{location}_train_eval_loader_{self.forecast_lead_hours}h_forecast_{target_variable}.pth"

        # Create folders in which to save files
        os.makedirs(self.images_output_path, exist_ok=True)
        os.makedirs(self.csv_output_path, exist_ok=True)

        # Models parameters
        self.years_of_data_to_use = float(os.getenv("YEARS_TO_USE"))
        self.learning_rate = float(os.getenv("LEARNING_RATE"))
        self.num_layers = int(os.getenv("NUM_LAYERS"))
        self.num_hidden_units = int(os.getenv("NUM_HIDDEN_UNITS"))
        self.train_test_ratio = float(os.getenv("TRAIN_TEST_RATIO"))
        self.epochs = int(os.getenv("EPOCHS"))
        self.batch_size = int(os.getenv("BATCH_SIZE"))
        self.dropout = float(os.getenv("DROPOUT"))
        self.normalization_method = os.getenv("NORMALIZATION_METHOD")
        self.weight_decay = float(os.getenv("WEIGHT_DECAY"))

        self.seq_length = round(self.forecast_lead_hours // float(os.getenv("SEQ_LENGTH_DIVIDER")))
        self.loss_function = None
        self.optimizer = None
        self.features = None

        # Loading existing models and loaders
        self.model = torch.load(self.model_path) if os.path.exists(self.model_path) else None
        self.train_loader = torch.load(self.train_loader_path) if os.path.exists(self.model_path) else None
        self.test_loader = torch.load(self.test_loader_path) if os.path.exists(self.model_path) else None
        self.train_eval_loader = torch.load(self.train_eval_loader_path) if os.path.exists(self.model_path) else None

        # Counter for training epochs
        self.training_counter = 0

    def build_new_model(self):
        print(f"Starting run -" + f"location: {self.location}\tvariable: {self.target_variable}\t"
                                  f"forecast hours: {self.forecast_lead_hours}\t" +
              f"\nParameters of this run:\nyear of data:{self.years_of_data_to_use}\ttrain ratio:{self.train_test_ratio}"
              + f"\nlearning rate: {self.learning_rate}\t"
                f"num_layers: {self.num_layers}\tnum_hidden_units: {self.num_hidden_units}\tbatch size: {self.batch_size}"
              + f"\nsequence length: {self.seq_length}\n timestamp of run: {datetime.datetime.now()}\n")
        # Load data
        print(f"###########\t\tCreating model for forecast lead:  {self.forecast_lead_hours} hours")

        forecast_lead = self.forecast_lead_hours * 2  # Since rows are for each 30 min, not 1 hour.
        self.full_data_df = create_short_data_csv(self.full_data_path,self.wind_data_path,self.swell_data_path,self.ims_data_path,
                                                  self.new_data_training_path,
                                                  self.forecast_groundtruth, forecast_lead,
                                                  self.seq_length, years_of_data_to_use=self.years_of_data_to_use)
        full_data_df, self.features, new_target = load_data(data=self.full_data_df,
                                                            features_mask='all',
                                                            forecast_lead=forecast_lead,
                                                            target_variable=self.target_variable)
        df_train, df_test = train_test_split(full_data_df, ratio=self.train_test_ratio)

        train_dataset = SequenceDataset(df_train, new_target, self.features, self.seq_length)
        test_dataset = SequenceDataset(df_test, new_target, self.features, self.seq_length)
        train_dataset.normalize_features_and_target(method=self.normalization_method)
        test_dataset.columns_std = copy.deepcopy(train_dataset.columns_std)
        test_dataset.columns_mean = copy.deepcopy(train_dataset.columns_mean)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.train_eval_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        torch.save(self.train_loader, self.train_loader_path)
        torch.save(self.test_loader, self.test_loader_path)
        torch.save(self.train_eval_loader, self.train_eval_loader_path)

        # train model
        self.model, self.loss_function, self.optimizer = configure_new_model(self.features, self.learning_rate,
                                                                             self.num_hidden_units, self.num_layers,
                                                                             self.dropout, self.weight_decay,
                                                                             self.target_variable)
        return self.train_lstm()

    def predict_latest_available_data(self):
        if None in [self.model, self.train_loader, self.test_loader, self.train_eval_loader]:
            print("ERROR: one of the files was not found. exiting")
            print(f"model={self.model}\ntrain_loader={self.train_loader}\ntest_loader={self.test_loader}\n"
                  f"train_eval_loader={self.train_eval_loader}")
            return

        # Need to put the new data in a loader including normalization and cleaning.
        train_seq_dataset = self.train_loader.dataset
        normalization_dicts = train_seq_dataset.get_normalization_dicts()
        self.features = train_seq_dataset.features
        new_data_df = create_short_data_csv(self.full_data_path, self.wind_data_path,self.swell_data_path,self.ims_data_path,
                                            self.new_data_training_path, self.forecast_groundtruth, self.forecast_lead_hours * 2,
                                            self.seq_length, predict_latest=True)
        prediction_df, _, new_target = load_data(data=new_data_df,
                                                 features_mask=self.features,
                                                 forecast_lead=self.forecast_lead_hours * 2,
                                                 target_variable=self.target_variable,
                                                 new_data=True)
        new_dataset = SequenceDataset(prediction_df, new_target, self.features, self.seq_length)
        new_dataset.normalize_features_and_target(imported_dicts=normalization_dicts, method=self.normalization_method)
        new_loader = DataLoader(new_dataset, batch_size=self.batch_size, shuffle=False)
        predictions_column_name = f"predicted_{self.target_variable}"

        prediction_df[predictions_column_name] = train_seq_dataset.invert_normalization(self.model.predict(new_loader), self.target_variable)

        # Smoothing the values
        if self.target_variable == 'tp':
            prediction_df[predictions_column_name] = prediction_df[predictions_column_name]
        else:
            prediction_df[predictions_column_name] = prediction_df[predictions_column_name].rolling(window=5,
                                                                                                    min_periods=1).mean()

        prediction_df.index = pd.to_datetime(prediction_df.index)
        forcast_start_datetime = prediction_df.index[-1 * self.forecast_lead_hours * 2]
        predicted_values_only = prediction_df.loc[forcast_start_datetime:, predictions_column_name]
        predicted_values_only = predicted_values_only.reset_index()
        predicted_values_only['datetime'] = predicted_values_only['datetime'].apply(lambda x: x + relativedelta(
            hours=self.forecast_lead_hours))

        forcast_start_datetime_string = forcast_start_datetime.strftime("%Y-%m-%d_%H_%M_%S")
        predicted_values_only.to_csv(
            f"{self.csv_output_path}/forcast_{self.forecast_lead_hours}h_start_time_{forcast_start_datetime_string}.csv",
            index=False)
        utils.upload_predictions_to_db(predicted_values_only.values, location=self.location,
                                       target_variable=self.target_variable)

    def predict_on_new_data_csv(self, data_to_predict_csv_path):
        train_seq_dataset = self.train_loader.dataset
        normalization_dicts = train_seq_dataset.get_normalization_dicts()
        self.features = train_seq_dataset.features
        # Need to put the new data in a loader.
        prediction_df, _, new_target = load_data(data=data_to_predict_csv_path,
                                                 features_mask=self.features,
                                                 forecast_lead=self.forecast_lead_hours * 2,
                                                 target_variable=self.target_variable,
                                                 new_data=True)
        new_dataset = SequenceDataset(prediction_df, new_target, self.features, self.seq_length)
        new_dataset.normalize_features_and_target(imported_dicts=normalization_dicts, method=self.normalization_method)
        new_loader = DataLoader(new_dataset, batch_size=self.batch_size, shuffle=False)

        predictions_column_name = 'Model Prediction'
        prediction_df[predictions_column_name] = train_seq_dataset.invert_normalization(self.model.predict(new_loader), column=self.target_variable)
        prediction_df[self.target_variable] = train_seq_dataset.invert_normalization(prediction_df[self.target_variable], column=self.target_variable)
        prediction_df = prediction_df.loc[:, [self.target_variable, predictions_column_name]]
        forcast_start_datetime = prediction_df.index[-1 * self.forecast_lead_hours * 2]
        predicted_values_only = prediction_df.loc[forcast_start_datetime:, predictions_column_name]
        predicted_values_only = predicted_values_only.reset_index()
        predicted_values_only['datetime'] = pd.to_datetime(predicted_values_only['datetime'])
        predicted_values_only['datetime'] = predicted_values_only['datetime'].apply(lambda x: x + relativedelta(
            hours=self.forecast_lead_hours - 2))
        predicted_values_only['smoothed_values'] = predicted_values_only[predictions_column_name].rolling(window=5, min_periods=1).mean()
        predicted_values_only.to_csv(f"{self.csv_output_path}/new_data_predictions_forcast_only.csv", index=False)

        prediction_df.to_csv(f"{self.csv_output_path}/new_data_predictions_full.csv")
        predicted_values_only.to_csv(f"{self.csv_output_path}/new_data_predictions_forcast_only.csv", index=False)
        # compare with real results

        empirical_measurments = pd.read_csv(self.forecast_groundtruth, index_col="datetime")[
                                :len(predicted_values_only)]
        empirical_measurments[predictions_column_name] = predicted_values_only.loc[:len(empirical_measurments)][
            predictions_column_name].values
        empirical_measurments['smoothed_values'] = predicted_values_only.loc[:len(empirical_measurments)][
        'smoothed_values'].values
        x_axis_labels = [(x+relativedelta(hours=3)).strftime("%H:%M") for x in predicted_values_only['datetime']]
        plt.plot(predicted_values_only['datetime'], empirical_measurments[predictions_column_name], label="Predictions")
        plt.plot(predicted_values_only['datetime'], empirical_measurments[self.target_variable], label="Real Data")
        plt.plot(predicted_values_only['datetime'], empirical_measurments['smoothed_values'], label="Smoothed Predictions")
        plt.xticks(labels=x_axis_labels, rotation=90, ticks=predicted_values_only['datetime'], )
        plt.rc('xtick', labelsize=7)
        plt.title(f'Start: {forcast_start_datetime} : Forecast vs Real Data epoch {self.training_counter}')
        plt.legend()
        output_image_path = os.path.join(self.images_output_path,
                                         f"forecast_predictions_epoch_{self.training_counter}.png")
        plt.savefig(output_image_path)
        plt.close()

    def train_lstm(self):
        print("Untrained test\n--------")
        self.model.test_model(self.test_loader, self.loss_function)
        train_loss = []
        test_loss = []
        for ix_epoch in tqdm.tqdm(range(self.epochs)):
            print(f"Epoch {ix_epoch}\n---------")
            self.model, train_loss_epoch = self.model.train_epoch(self.train_loader, self.loss_function, self.optimizer)
            test_loss_epoch = self.model.test_model(self.test_loader, self.loss_function)
            torch.save(self.model, self.model_path)
            self.predict_on_new_data_csv(self.new_data_training_path)
            self.training_counter += 1
            if ix_epoch == 0:
                continue
            train_loss.append(train_loss_epoch)
            test_loss.append(test_loss_epoch)
        plt.plot(range(self.epochs - 1), train_loss, label="Train Loss")
        plt.plot(range(self.epochs - 1), test_loss, label="Test Loss")
        plt.title('Train/Test Loss')
        plt.legend()
        plt.savefig(os.path.join(self.images_output_path, f"loss_graphs.png"))
        plt.close()
        print(f"Train loss:\n{train_loss}")
        print(f"Test loss:\n{test_loss}")
        return train_loss, test_loss
