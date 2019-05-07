import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings; warnings.simplefilter('ignore')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, precision_score, recall_score, log_loss, roc_auc_score, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble.partial_dependence import partial_dependence


class ReadMyFiles():
    ''' These will format BigCommerce output files specifically :) '''

    def __init__(self):

        self.custy_df = None
        self.order_df = None
        self.subscriber_df = None
        self.product_df = None

    def read_customer(self, filepath):

        custy_df = pd.read_csv(filepath, skiprows=None, index_col='Customer ID', parse_dates=["Date Joined"]).dropna(axis=0, how='all')
        self.custy_df = custy_df.drop(columns=['Rep Name', 'Education'], axis=1)
        return self.custy_df
    
    def read_order(self, filepath):

        self.order_df = pd.read_csv(filepath, parse_dates=["Order Date"])
        return self.order_df

    def read_product(self, filepath):

        self.product_df = pd.read_csv(filepath)
        return self.product_df

    def read_marketing(self, filepath):

        self.subscriber_df= pd.read_csv(filepath)
        return self.subscriber_df

def assemble_cold_start_feature_df(custy_df, order_df, subscriber_df):
        
        cold_start_feature_extraction_df = pd.DataFrame(columns=['order_value', 'time_as_customer', 'ship',
                                                    'avg_price_item', 'subscriber_newsletter', 'uses_coupons'])

        cold_start_feature_extraction_df = cold_start_feature_extraction_df.reindex(custy_df.index)

        subscriber_emails = subscriber_df['Email'].astype(list)

        for customer in custy_df.index.values:


            mask = order_df[order_df['Customer ID'] == customer] # mask for all orders under a customer
            if len(mask) == 0: # skip this customer if they've never ordered
                pass

            else:

                # time as customer
                time_as_customer = (mask["Order Date"].max() - custy_df['Date Joined'][customer]).days

                ### GET COLD START
                ''' What happens when we have a brand new customer?
                We only have 'avg_order_value', 'avg_price_item', 'subscriber_newsletter', 'uses_coupons', 'customer_group' and 'affiliation' 
                I've truncated everybody's orders after their first order -- can we predict how long they will 'be customers'? time_as_customer is defined
                as time between date_joined and last order date (true) '''

                mask_zero = mask.head(1)

                # average order value
                order_value = mask_zero['Subtotal'].sum()

                # avg $ spent on shipping
                ship = round(mask_zero['Shipping Cost'].sum(), 3)

                # average price per items purchased
                avg_price_per_item = mask_zero['Subtotal'].sum()/mask_zero['Total Quantity'].sum()

                # is subscriber
                if mask_zero['Customer Email'].values[0] in subscriber_emails: 
                    is_subscriber = 1
                else: is_subscriber = 0

                # uses coupons
                coupons_list = [x for x in list(mask_zero['Coupon Details']) if str(x) != 'nan']
                if len(coupons_list) > 0:
                    uses_coupons = 1
                else: uses_coupons = 0

                cold_start_feature_extraction_df.loc[customer]['order_value'] = order_value
                cold_start_feature_extraction_df.loc[customer]['time_as_customer'] = time_as_customer
                cold_start_feature_extraction_df.loc[customer]['avg_price_item'] = avg_price_per_item
                cold_start_feature_extraction_df.loc[customer]['subscriber_newsletter'] = is_subscriber
                cold_start_feature_extraction_df.loc[customer]['uses_coupons'] = uses_coupons
                cold_start_feature_extraction_df.loc[customer]['ship'] = ship


        cold_start_feature_extraction_df['customer_group'] = custy_df['Customer Group']
        cold_start_feature_extraction_df['affiliation'] = custy_df['Your Affiliation']
        cold_start_feature_extraction_df.dropna(thresh=3, inplace=True)

        print("initial feature extractions completed for cold start.")

        return cold_start_feature_extraction_df

class Transform():
    ''' Transform initial feature dataframe into a binarized dataframe with all the
    price features logged and a churn boolean added
     
     Parameters
     ----------
    feature_df = the results of 'assemble_feature_matrix', should contain 1 column for 'Affiliation' and 1 column for 'Customer Group' 
    (these two are the only string cols, the rest are numeric)
     
     Attributes
     ----------
     this class uses one hot encoding, logs cost features, makes a churn boolean, and trims the dataframe
     option to fit an initial random forest to these features
     '''

    def __init__(self, feature_df):

        self.feature_df = feature_df

    def binarize(self):
        
        dummy_Cgroup = pd.get_dummies(self.feature_df['customer_group'])
        dummy_aff = pd.get_dummies(self.feature_df['affiliation'])
    
        dummy_merged = dummy_aff.merge(dummy_Cgroup, on='Customer ID')
        self.feature_df = self.feature_df.merge(dummy_merged, on='Customer ID')

    def transform_cold_start_data(self):

        self.feature_df['order_value_logged'] = self.feature_df['order_value'].apply(lambda x: np.log(x) if x > 0 else 0)
        self.feature_df['avg_price_item_logged'] = self.feature_df['avg_price_item'].apply(lambda x: np.log(x) if x > 0 else 0)
        self.feature_df['ship_logged'] = self.feature_df['ship'].apply(lambda x: np.log(x) if x > 0 else 0)
        self.feature_df = self.feature_df.drop(columns=['order_value', 'ship', 'customer_group', 'affiliation', 'avg_price_item'], axis=1)
        
        return self.feature_df
    

class Splitter():

    def __init__(self):

        self.feature_df = None

    def split_for_cold_start(self, feature_df):

        ''' What happens when we have a brand new customer?
        We have: id, CGroup, Affiliation, AOV, is_subscriber, avg_price_item at minimum
        We can do NMF and identify that customer with groups/products more likely to churn'''

        y = feature_df['time_as_customer']
        X = feature_df.drop(columns=['time_as_customer'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        return X_train, X_test, y_train, y_test

class Model_Cold_Start():
    ''' Use sklearn's implementation of NMF combined with some supervised learning techniques.
     
     Parameters
     ----------

     
     Attributes
     ----------
     '''
    def __init__(self):
        
        self.X = None
        self.y = None
        self.model = None
        self.y_pred = None  
        self.baseline = None

    def fit(self, X, y):
        
        self.X = X
        self.y = y

        self.baseline = self.y.mean()
        
        model = GradientBoostingRegressor(n_estimators = 150, learning_rate = 0.05, 
                                   max_features = 'sqrt', max_depth = 10)
        self.model = model.fit(self.X, self.y)
        print('Model fitted.')

    def predict(self, X):

        self.y_pred = self.model.predict(X)
        return self.y_pred

    def score(self, X_test, y_test):

        self.baseline = [self.baseline] * len(y_test)

        baseline_MSE = np.sqrt(mean_squared_error(y_test, self.baseline))
        print('Baseline Root Mean Squared Error:', round(baseline_MSE , 0))
        
        model_MSE = np.sqrt(mean_squared_error(y_test, self.y_pred))
        print('Model Root Mean Squared Error:', round(model_MSE, 0))

        print('Reduction in RMSE: ', ((model_MSE - baseline_MSE) / baseline_MSE))
        
        ### PD PLOT
        importances = self.model.feature_importances_
        sorted_imps = sorted(importances)[::-1]
        indicies = np.argsort(importances)[::-1]
        names = self.X.columns[indicies]
        N_COLS = 3

        pd_plots = [partial_dependence(self.model, target_feature, X=self.X, grid_resolution=50)
                    for target_feature in indicies]
        pd_plots = list(zip(([pdp[0][0] for pdp in pd_plots]), ([pdp[1][0] for pdp in pd_plots])))

        fig, axes = plt.subplots(nrows=3, ncols=N_COLS, sharey=True, 
                                figsize=(12.0, 8.0))

        for i, (y_axis, x_axis) in enumerate(pd_plots[0:(3*N_COLS)]):
            ax = axes[i//N_COLS, i%N_COLS]
            ax.plot(x_axis, y_axis, color="purple")
            ax.set_xlim([np.min(x_axis), np.max(x_axis)])
            text_x_pos = np.min(x_axis) + 0.05*(np.max(x_axis) - np.min(x_axis))
            ax.text(text_x_pos, 7.5,
                    "Feature Importance " + str(round(sorted_imps[i],2)), 
                    fontsize=12, alpha=0.7)
            ax.set_xlabel(names[i])
            ax.grid()
            
        plt.suptitle("Partial Dependence Plots (Ordered by Feature Importance)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


if __name__ == '__main__':

    ### LOAD MY FILES
    readme = ReadMyFiles()
    custy_df = readme.read_customer('custy_data.csv')
    order_df = readme.read_order('order_data.csv')
    subscriber_df = readme.read_marketing('subscribers-2019-03-27.csv')
    print('files read.')

    ### INITIAL EXTRACTION
    cold_start_feature_df = assemble_cold_start_feature_df(custy_df, order_df, subscriber_df)

    ### TRANSFORM DATA
    coldtransform = Transform(cold_start_feature_df)
    coldtransform.binarize()
    transformed = coldtransform.transform_cold_start_data()

    ### SPLITTER
    makesplits = Splitter()
    X_train, X_test, y_train, y_test= makesplits.split_for_cold_start(transformed)

    ### MODEL DATA
    coldModel = Model_Cold_Start()
    coldModel.fit(X_train, y_train)
    coldModel.predict(X_test)
    coldModel.score(X_test, y_test)
    



        