import pandas as pd
import datetime


def predict(predictor, data_train, y, data_test, cv_score=None):
    predictor.fit(data_train, y)
    prediction = predictor.predict(data_test)
    print predictor
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filepath_prediction = 'data/predictions/prediction-%s-data.csv' % timestamp
    filepath_description = 'data/predictions/prediction-%s-estimator.txt' % timestamp

    # Create a dataframe with predictions and write it to CSV file
    predictions_df = pd.DataFrame(data=prediction, index=data_test.index, columns=['Outcome'])
    predictions_df.to_csv(filepath_prediction, sep=',')

    # Write a short description of the classifier that was used
    f = open(filepath_description, 'w')
    f.write(str(predictor))
    f.write('\nCross-validation score %.8f' % cv_score)
    f.close()
