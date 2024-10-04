import pandas as pd
from utils.loaddata import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import h2o
from h2o.automl import H2OAutoML
import datetime
from pathlib import Path


class Trainer():
    def __init__(self, args) -> None:
        self.args = args
        

    def processData(self, data, correlation):
        print("processing training data")
        df= pd.read_csv(data)
        
        data = filter_data(df, workload = "", isbatchThroughput=False, custom_col_exclude=self.args.custom_col_exclude, n_combination=self.args.n_combination)
        #categorical_columns = ['workload1', 'workload2', 'idx1', 'idx2']
        
        #data, columns_excluded = preprocess_data(data=data, categorical_columns=categorical_columns, target='sum_throughput', correlation=0.2)
        #print(f"columns excluded {columns_excluded}")

        self.X_train, self.X_test, self.y_train, self.y_test, self.test_workloads, columns_excluded = train_test_custom_split(data=data,
                                                                                    test_workload="", 
                                                                                    target="sum_throughput", 
                                                                                    RANDOMSEED=30, CORRELATION=correlation,
                                                                                    n_combination=self.args.n_combination
                                                                                    )
        
        now = datetime.datetime.now()
        #get day, month, year
        day = now.day
        month = now.month
        year = now.year
        #get time
        current_time = now.strftime("%H:%M:%S")
        self.timestamp = f"{day}-{month}-{year}_{current_time}"

        return  columns_excluded
    
    def train(self, modeltype):
        #get time
        start_time = datetime.datetime.now()
        if modeltype == "KACE":
            model = LinearRegression()
            model.fit(self.X_train, self.y_train)
        elif modeltype == "NN":
            import tensorflow as tf
            tf.random.set_seed(50)
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            # Define the model
            model = Sequential([
                Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='linear')  # Output layer for regression
            ])

            model.compile(optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_squared_error'])
            #add dropout layer
            model.add(Dropout(0.2))
            #apply earlystopping
            from tensorflow.keras.callbacks import EarlyStopping
            early_stopping = EarlyStopping(patience=40)
            # Train the model with seeds
            history = model.fit(self.X_train, self.y_train, epochs=250, validation_split=0.2, verbose=1, callbacks=[early_stopping])
                        
        elif modeltype == "hotcloud" or modeltype == "RF":
            model = RandomForestRegressor(random_state=42)
            #random search - random select from the combination of hyperparameters
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            # Number of features to consider at every split
            max_features = ['sqrt', 'log2']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
            model = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
            model.fit(self.X_train, self.y_train)

            

        elif modeltype == "AutoML":

            h2o.init()
            h2o_train = h2o.H2OFrame(pd.concat([self.X_train, self.y_train], axis=1))
            h2o_test = h2o.H2OFrame(pd.concat([self.X_test, self.y_test], axis=1))
            # Identify predictors and response
            x = h2o_train.columns
            y = "sum_throughput"
            x.remove(y)
            #train the model
            aml = H2OAutoML(max_models=30, seed=30, sort_metric='mse', max_runtime_secs=3*60)
            aml.train(x=x, y=y, training_frame=h2o_train)
            model = aml.leader

            model_path = h2o.save_model(model=model, path=f"{self.args.output_dir}", force=True)
            print(f"Best model saved to: {model_path}")

        else:
            raise ValueError("modeltype not supported")
        
        
        end_time = datetime.datetime.now()
        process_time = str(end_time-start_time)
        str_excol = "_".join(self.args.custom_col_exclude)

        #save end_time-start_time to a file
        import os# Get the parent directory from the output file path
        # Create parent directory if it doesn't exist
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(f'{self.args.output_dir}/{self.timestamp}_{modeltype}_model-corr{self.args.correlation}_datard{self.args.split_randomseed}_excol{str_excol}_train_time.txt', 'w') as f:
            f.write("Training time\n")
            f.write(f"{process_time}\n")
        
        if  modeltype != "AutoML":
            with open(f'{self.args.output_dir}/{self.timestamp}_{modeltype}_model-corr{self.args.correlation}_datard{self.args.split_randomseed}_excol{str_excol}.pkl', 'wb') as file:
                pickle.dump(model, file)

        #load   the model from a file
        #with open(f'output/trained_models/{self.timestamp}linear_regression_model.pkl', 'rb') as file:
        #    model = pickle.load(file)
        if modeltype != "AutoML":
            y_pred = model.predict(self.X_test)
        else:
            y_pred = model.predict(h2o_test)
            y_pred = h2o.as_list(y_pred)
        
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f'MSE: {mse}')
        print(f'R^2: {r2}')
        
        #save mse,r2 to a csv file
        with open(f'{self.args.output_dir}/{self.timestamp}mse_r2-corr{self.args.correlation}_datard{self.args.split_randomseed}_excol{str_excol}.csv', 'w') as f:
            f.write("mse,r2\n")
            f.write(f"{mse},{r2}\n")

        return Path(f"{self.args.output_dir}/{self.timestamp}_{modeltype}_model-corr{self.args.correlation}_datard{self.args.split_randomseed}_excol{str_excol}.pkl")