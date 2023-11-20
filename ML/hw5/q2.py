import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def preprocess(df):
    categorical_columns = df.select_dtypes(include='object').columns

    numerical_columns = [col for col in df.columns if col not in categorical_columns]

    column_to_remove = 'class'

    updated_numerical_columns = [col for col in numerical_columns if col != column_to_remove]

    scaler = MinMaxScaler()
    df[updated_numerical_columns] = scaler.fit_transform(df[updated_numerical_columns])
    term_dummies = pd.get_dummies(df['term'], prefix='term')
    grade_dummies = pd.get_dummies(df['grade'], prefix='grade')  
    emp_length_dummies = pd.get_dummies(df['emp_length'], prefix='emp_length')
    home_own_dummies = pd.get_dummies(df['home_ownership'], prefix='home_ownership')
    verif_dummies = pd.get_dummies(df['verification_status'], prefix='verification_status')
    purpose_dummies = pd.get_dummies(df['purpose'], prefix='purpose')
    earliest_dummies = pd.get_dummies(df['earliest_cr_line'], prefix='earliest_cr_line' )
    df = pd.concat([df, term_dummies, grade_dummies, emp_length_dummies, home_own_dummies,
                    verif_dummies, purpose_dummies, earliest_dummies], axis=1)

    original_cols = ['term', 'grade', 'emp_length', 'home_ownership', 
                    'verification_status', 'purpose', 'earliest_cr_line']
    df.drop(original_cols, axis=1, inplace=True)

    df.rename(columns={
        'term_36 months':'term_36mo',
        'term_60 months':'term_60mo',
        'grade_A':'grade_A',
        'grade_B':'grade_B', 
        'grade_C':'grade_C',
        'grade_D':'grade_D', 
        'grade_E':'grade_E',
        'grade_F':'grade_F',
        'grade_G':'grade_G', 
        'emp_length_n/a':'emp_length_na',
        'emp_length_< 1 year':'emp_length_<1', 
        'emp_length_1 year':'emp_length_1',
        'emp_length_2 years':'emp_length_2', 
        'emp_length_3 years':'emp_length_3',
        'emp_length_4 years':'emp_length_4',
        'emp_length_5 years':'emp_length_5',
        'emp_length_6 years':'emp_length_6', 
        'emp_length_7 years':'emp_length_7',
        'emp_length_8 years':'emp_length_8',
        'emp_length_9 years':'emp_length_9',
        'emp_length_10+ years':'emp_length_10',
        'home_ownership_MORTGAGE':'home_MORTGAGE',
        'home_ownership_OTHER':'home_OTHER',  
        'home_ownership_OWN':'home_OWN',
        'home_ownership_RENT':'home_RENT',
        'verification_status_Not Verified':'verif_NotVerified',
        'verification_status_Source Verified':'verif_SourceVerified',
        'verification_status_Verified':'verif_Verified',
        'purpose_car':'purpose_car',
        'purpose_credit_card':'purpose_credit_card',  
        'purpose_debt_consolidation':'purpose_debt_consolidation',
        'purpose_educational':'purpose_educational',
        'purpose_home_improvement':'purpose_home_improvement',
        'purpose_house':'purpose_house',
        'purpose_major_purchase':'purpose_major_purchase',  
        'purpose_medical':'purpose_medical',
        'purpose_moving':'purpose_moving',
        'purpose_other':'purpose_other',
        'purpose_renewable_energy':'purpose_renewable_energy',
        'purpose_small_business':'purpose_small_business',
        'purpose_vacation':'purpose_vacation',
        'purpose_wedding':'purpose_wedding',
    }, inplace=True)

    data = df.to_numpy()
    return data

def run_pca(train_x, test_x):
   
    scaler = StandardScaler()
    scaler.fit(train_x) 
    X_train_scaled = scaler.transform(train_x)

    X_test_scaled = scaler.transform(test_x) 

    pca = PCA()
    pca.fit(X_train_scaled)

    total_variance = np.sum(pca.explained_variance_ratio_)
    cumulative_variance = 0
    num_components = 0
    for explained_variance in pca.explained_variance_ratio_:
        cumulative_variance += explained_variance
        num_components += 1
        if cumulative_variance >= 0.95 * total_variance:
            break
    train_loadings = pca.transform(X_train_scaled)[:,:num_components]
    test_loadings = pca.transform(X_test_scaled)[:,:num_components]
    principal_components = pca.components_[:num_components,:]
 
    return (num_components,
            principal_components,
            train_loadings,
            test_loadings)

def run_nmf(train_x, test_x, num_components):
    
    # scaler = MinMaxScaler()
    # scaled_train_x = scaler.fit_transform(train_x)
    # scaled_test_x = scaler.transform(test_x)
    
    nmf = NMF(n_components=num_components)
    
    nmf.fit(train_x)
    
    train_error = nmf.reconstruction_err_
    train_features = nmf.transform(train_x)
    test_features = nmf.transform(test_x)
    components = nmf.components_
    
    return (train_error, components, train_features, test_features)