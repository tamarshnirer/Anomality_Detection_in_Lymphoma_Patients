from sklearn.base import TransformerMixin
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sparse_hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
import numpy as np
import pandas as pd


class Initial_pre_processing_Transformer(TransformerMixin):
    '''
    Performs initial preprocessing on the input DataFrame:
    1) Dropping irrelevant columns for prediction
    2) Mapping the following columns to Boolean(1/0): "Gender", "Included in Survival Analysis", "Number of Extranodal Sites" -> "Binary_num_of_nodes" if number is greater than zero.
    Parameters
    ----------
    None
    Returns
    ----------
    DataFrame
    Returns the input df after initial preprocessing
    '''   
#     X_cols = []
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        '''
        Returns Self
        Parameters
        ----------
        None
        Returns
        ----------
        self
        '''   
        return self
    
    def transform(self,X):
        '''
        Performs all initial preprocessing:
            1) Dropping irrelevant columns for prediction
            2) Mapping the following columns to Boolean(1/0): "Gender", "Included in Survival Analysis", "Number of Extranodal Sites" -> "Binary_num_of_nodes" if number is greater than zero.
        Parameters
        ----------
        None
        Returns
        ----------
         DataFrame
        Returns the input df after initial preprocessing
        '''   
        X_copy = X.copy()
        X_copy.drop(X_copy.columns[[0, 2, 19]], axis=1, inplace = True) #dropping the diagnosis (the same for all) and other two feature that might predict what we're trying to predict 
        X_copy['Gender'] = X_copy['Gender'].map({'M': 0,'F': 1})
#         X_copy['Included in Survival Analysis'] = X_copy['Included in Survival Analysis'].map({'Yes': 1,'No': 0})
        X_copy.replace("Pre-treatment", "Pretreatment",inplace=True)
        X_copy['binary_num_of_nodes'] = X_copy['Number of Extranodal Sites'].copy()
                
        # Change only the not-null values
        not_null_node_indices = X_copy[~X_copy['Number of Extranodal Sites'].isna()].index
        X_copy.loc[not_null_node_indices,'binary_num_of_nodes'] = np.where(X_copy.loc[not_null_node_indices,:]['Number of Extranodal Sites']>0, 1, 0)

        return X_copy.drop(['binary_num_of_nodes','Number of Extranodal Sites'], axis =1) , X_copy['binary_num_of_nodes']   
    
class FeatureTransformer(TransformerMixin):
    def __init__(self,  is_numeric_Transformer=True, is_cat_Transformer=True, is_ipi_values_Transformer=True
                ):
        '''
        Consists of 3 transformers: 1) numeric_Transformer, 2) cat_Transformer, 3) ipi_values_Transformer, 
        
        Parameters
        ----------
        1) is_numeric_Transformer - True if we want to enable the numeric_Transformer, Otherwise False
        2) is_cat_Transformer - True if we want to enable the cat_Transformer, Otherwise False
        3) is_ipi_values_Transformer - True if we want to enable the ipi_values_Transformer, Otherwise False
        '''
        feature_for_transformer =[]

        if is_numeric_Transformer:
            feature_for_transformer.append(('num',numeric_Transformer()))    
            
        if is_cat_Transformer:
            feature_for_transformer.append(('',cat_Transformer()))
        
        if is_ipi_values_Transformer:
            feature_for_transformer.append(('ipi_values',ipi_values_Transformer()))
        
        self.transformer = FeatureUnion(feature_for_transformer)
    
    def fit(self,X,y=None):
        '''
        Run each of the activated transformers' fit functions
        Parameters
        ----------
        None
        Returns
        ----------
        Self 
        '''
        return self.transformer.fit(X)
    
    def transform(self,X):
        '''
        Run each of the activated transformers' transform functions
        Parameters
        ----------
        None
        Returns
        ----------
        DataFrame
            DataFrame with all of the transformed columns
        '''        
        return self.transformer.transform(X).astype(float)
    
    def get_feature_names(self):
        '''
        Run each of the activated transformers' get_feature_names functions
        Parameters
        ----------
        None
        Returns
        ----------
        List
            A list of feature names
        '''   
        return self.transformer.get_feature_names()

#***************************************************************************************
class numeric_Transformer(TransformerMixin):
    '''
    Imputes all numeric columns with their medians on the training set.
    Parameters
    ----------
    None
    Returns
    ----------
    Sparse Matrix
            Matrix consisting of the numeric columns after imputation
    '''   
    numeric_features = ['Ann Arbor Stage','LDH Ratio','ECOG Performance Status','Gender','Age',
                       'Follow up Status Alive=0 Dead=1', 'Follow up Time (yrs)',
       'PFS Status No Progress=0 Progress=1', 'PFS (yrs)']
#        'Included in Survival Analysis']

    num_of_numeric_features = len(numeric_features)
    def __init__(self,num_medians=np.zeros(num_of_numeric_features), **cv_kwargs):
        self.num_medians = num_medians

        pass

    def fit(self, X, y=None):
        '''
        Learn the median values of the numeric columns
        Parameters
        ----------
        None
        Returns
        ----------
        self
        ''' 
        self.num_medians = X[self.numeric_features].median()
#         print(self.num_medians)
        return self
    
    def transform(self, X, y=None):
        '''
        Transforms all anon numeric to a sparse matrix
        Parameters
        ----------
        None
        Returns
        ----------
        Sparse Matrix
            Matrix consisting of all numeric features after imputation with median values
        '''   
        X_copy = X.copy()        

        X_num = csr_matrix(X_copy[self.numeric_features].fillna(self.num_medians).apply(pd.to_numeric,errors='coerce').values.astype(float))
        return sparse_hstack([X_num])
    
    def get_feature_names(self):
        '''
        returns a list of feature names consisting of each of the numeric_features.
        Parameters
        ----------
        None
        Returns
        ----------
        List
            A list of feature names
        '''   
        return [x.lower() for x in self.numeric_features]    
    
#***************************************************************************************

class cat_Transformer(TransformerMixin):
    '''
    Imputes all missing values in the category columns with 'unknown' and transforms the category_cols into OHE
    
    Parameters
    ----------
    None
    Returns
    ----------
    Sparse Matrix
            Matrix consisting of the numeric columns after imputation        
    '''   
    
    category_cols = ['Gene Expression Subgroup','Biopsy Type','Treatment', 'Genetic Subtype']
    
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        '''
        Learn a vocabulary dictionary of all tokens in the the category_cols.
        Parameters
        ----------
        None
        Returns
        ----------
        self
        '''   
        category_col_ohe_names = [col.lower() for col in self.category_cols]
        category_col_ohe = [OHEcol(col) for col in self.category_cols]
        self.category_feature_union = FeatureUnion([(category_col_ohe_names[i], category_col_ohe[i]) for i, _ in enumerate(category_col_ohe_names)])
        self.category_feature_union.fit(X[self.category_cols])

        return self
    
    def transform(self,X):
        '''
        Transforms category columns to a sparse matrix
        Parameters
        ----------
        None
        Returns
        ----------
        Sparse Matrix
            Matrix consisting of the OHE columns
        '''   
        X_copy = X.copy()        
        X_one_hot_category = self.category_feature_union.transform(X[self.category_cols].fillna(''))
        return sparse_hstack([X_one_hot_category])

    def get_feature_names(self):
        '''
        returns a list of feature names consisting of each of the category cols.
        Parameters
        ----------
        None
        Returns
        ----------
        List
            A list of feature names
        '''   

        X_one_hot_category = self.category_feature_union.get_feature_names()
        return X_one_hot_category
    
class OHEcol(TransformerMixin):
    '''
    This transformer receives as an input one category column at each time. Imputes missing values with 'unknown' and applies a CountVectorizer on the column.
    Parameters
    ----------
    None
    Returns
    ----------
    Sparse Matrix
            Matrix consisting of the OHE of the column
    '''
    def __init__(self,col):
        self.col = col
        self.cv = CountVectorizer(min_df=0.05)
        return
    
    def _prepare(self,X):
        '''
        Imputes missing values with 'unknown'
        Parameters
        ----------
        None
        Returns
        ----------
        Series
            Returns the column after imputation with 'unknown'
        '''  
        return X[self.col].fillna('Unknown')  
    
    def fit(self, X, y=None):
        '''
        Learn a vocabulary dictionary of all tokens in the OHE col.
        Parameters
        ----------
        None
        Returns
        ----------
        self
        ''' 
        self.cv.fit(self._prepare(X).astype(str))
        return self
    
    def transform(self, X, y=None):
        '''
        Creates a OHE of the column using CountVectorizer
        Parameters
        ----------
        None
        Returns
        ----------
        Sparse Matrix
            Matrix consisting a OHE for the column values that appeared in at least 0.05 of the data (min_df=0.05)
        '''   
        return self.cv.transform(self._prepare(X).astype(str))
    
    def get_feature_names(self):
        '''
        returns a list of feature names consisting of each of the OHE cols.
        Parameters
        ----------
        None
        Returns
        ----------
        List
            A list of feature names
        '''   
        return [cv_f for cv_f in self.cv.get_feature_names()]

#***************************************************************************************

class ipi_values_Transformer(TransformerMixin):
    
    '''
    This transformer gets as an input a dataframe with the following columns: 1) 'IPI Group', 2) 'IPI Range' and performs the following:
    1) Imputes missing 'IPI Group' values based on the range, based on the logic - Low (0-1), Intermediate (2-3), High (4-5)
    2) Calculates mean, min and max IPI Values
    
    Parameters
    ----------
    None
    Returns
    ----------
    Sparse Matrix
            Matrix consisting of the new calculated columns        
    '''  
    
    def __init__(self):
        pass



    def fit(self, X, y=None):
        '''
        Returns self
        Parameters
        ----------
        None
        Returns
        ----------
        self
        ''' 
        return self
    
    def transform(self, X, y=None):
        '''
        1) Imputes missing 'IPI Group' values based on the range, based on the logic - Low (0-1), Intermediate (2-3), High (4-5)
        2) Calculates mean, min and max IPI Values
        Parameters
        ----------
        None
        Returns
        ----------
        Sparse Matrix
            Matrix consisting of the numeric columns, the numeric_fill_na after imputation with 0, num_of_guest collumn and OHE columns
        '''   
        
        def impute_ipi(row):
            ipi_group = row['IPI Group']
            ipi_range = row['IPI Range']
            if pd.isna(ipi_group):                
                
                # calculate mean from ipi_range 
                if ipi_range>5:
                    digits = [int(d) for d in str(ipi_range)]
                    ipi_range =  sum(digits)/len(digits)
                
                if 0<=ipi_range<2:
                    return 1  # 'Low'
                elif 2<=ipi_range<4:
                    return 2  # 'Intermediate'
                else:
                    return 3  # 'High'
            else:
                if ipi_group =='Low':
                    return 1 # 'Low'
                elif ipi_group =='Intermediate':
                    return 2 # 'Intermediate'
                else:
                    return 3
                    
        def get_mean_ipi(number):
            if number>5:
                digits = [int(d) for d in str(number)]
                return sum(digits)/len(digits)
            else:
                return number

        def get_min_ipi(number):
            if number>5:
                digits = [int(d) for d in str(number)]
                return min(digits)
            else:
                return number

        def get_max_ipi(number):
            if number>5:
                digits = [int(d) for d in str(number)]
                return max(digits)
            else:
                return number    
            
        X_mean = csr_matrix(X['IPI Range'].apply(lambda row: get_mean_ipi(row)).values).T
        X_min = csr_matrix(X['IPI Range'].apply(lambda row: get_min_ipi(row)).values).T
        X_max = csr_matrix(X['IPI Range'].apply(lambda row: get_max_ipi(row)).values).T
        ipi_group = csr_matrix(X[['IPI Group','IPI Range']].apply(lambda row: impute_ipi(row), axis = 1).values).T
        return sparse_hstack([X_mean,X_min,X_max,ipi_group])

   
    
    def get_feature_names(self):
        '''
        returns the relevant_col name.
        Parameters
        ----------
        None
        Returns
        ----------
        String
            relevant_col
        '''   
        return ['mean','min','max','ipi_group']