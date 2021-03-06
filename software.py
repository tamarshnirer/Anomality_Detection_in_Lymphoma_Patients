from tkinter import *
import pandas as pd
import pickle
from main import Initial_pre_processing_Transformer, FeatureTransformer, numeric_Transformer, cat_Transformer, OHEcol, ipi_values_Transformer
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


with open('finalized_RF_model.sav', 'rb') as f:
    rf = pickle.load(f)
with open('feature_transformer.pkl', 'rb') as f:
    transformer = pickle.load(f)
with open('minmaxScalerX.pkl', 'rb') as f:
    scaler = pickle.load(f)

Initial_pre_processing_Transformer = Initial_pre_processing_Transformer()
feat_group = ['num__ann arbor stage', '__gene expression subgroup__gcb',\
       'ipi_values__min', 'ipi_values__max',\
       '__gene expression subgroup__unclass',\
       'num__ecog performance status', '__genetic subtype__ezb',\
       '__genetic subtype__bn2', 'num__ldh ratio',\
       '__genetic subtype__mcd', 'ipi_values__mean',\
       'num__follow up status alive=0 dead=1', 'num__gender',\
       '__gene expression subgroup__abc', 'num__pfs (yrs)']

def compute_results():
    global Age
    Age = int((globals()['button Age']).get())
    global Ann_arbor_stage
    Ann_arbor_stage = int((globals()['button Ann Arbor Stage (0-5)']).get())
    global LDH_Ratio
    LDH_Ratio = float((globals()['button LDH Ratio (mkat/L)']).get())
    global ECOG
    ECOG = int((globals()['button ECOG Performance Status (0-4)']).get())
    global IPI_Range
    IPI_Range = int((globals()['button IPI Range (0-5)']).get())
    if Treatment.get() == 'Unknown':
        Treatment2 = np.nan
    else:
        Treatment2 = Treatment.get()

    vector = pd.DataFrame({'dbGaP submitted subject ID': 0,'dbGaP accession': 0 ,'Diagnosis': 0 ,'Gene Expression Subgroup': Gene_Expression_Subgroup.get(),'Genetic Subtype': Biopsy_type.get(),'Biopsy Type' :Biopsy_type.get(),'Treatment': Treatment2,'Gender': Gender.get(),'Age': Age,'Ann Arbor Stage': Ann_arbor_stage,'LDH Ratio': LDH_Ratio,'ECOG Performance Status': ECOG,'Number of Extranodal Sites': 0,'IPI Group': IPI_Group.get(),'IPI Range': IPI_Range,'Follow up Status Alive=0 Dead=1': np.nan,'Follow up Time (yrs)': np.nan,'PFS Status No Progress=0 Progress=1': np.nan,'PFS (yrs)': np.nan,'Included in Survival Analysis': np.nan}, index=[0])
    df_after_preprocessing, y = Initial_pre_processing_Transformer.fit_transform(vector)
    x_transformed = pd.DataFrame(transformer.transform(df_after_preprocessing).A,
                                 index=df_after_preprocessing.index, columns=transformer.get_feature_names())
    X = pd.DataFrame(scaler.transform(x_transformed),
                                   columns=x_transformed.columns)
    pred = rf.predict_proba(X[feat_group])
    output = f"This patient {round(pred[0,1],4)} chances of having extanodal site\s"

    Label(root, text = output).grid(row=41, column=1)


root = Tk()
root.title("Lymphoma Prediction System")

label2 = Label(root, text = "Please fill in the following fields:").grid(row=0, column=0)

row = 6
''''
col_list = ['Ann Arbor Stage (0-5)', 'Gene Expression Subgroup (Unclass,GCB,ABC)', 'Age', 'Gender (M,F)', 'LDH Ratio (mkat/L)', 'Ecog Performance Status (0-4)', 'Follow Up Status (alive=0, dead=1)', 'Follow Up Time (years)',\
            'PFS Status (no progress=0, progress=1)','PFS (years)', 'Included in Survival Analysis (yes=1, no=0)','Biopsy type (Pre-treatment, Relapse)', 'Treatment (Immunochemotherapy, Ibrutinib monotherapy)', 'Genetic Subtype (Other,EZB,BN2,N1,MCD)',\
            'IPI Group (Low, Intermediate, High)', 'IPI Range (0-5)']
'''
numeric_coluns = ['Ann Arbor Stage (0-5)', 'Age',  'LDH Ratio (mkat/L)', 'ECOG Performance Status (0-4)', 'IPI Range (0-5)']



Gender = StringVar(root)
Gender1 = OptionMenu(root, Gender, "M", "F").grid(row=30,column=1)
Label(root, text= "Gender:").grid(row=30,column=0)



Gene_Expression_Subgroup = StringVar(root)
Gene_Expression_Subgroup1 = OptionMenu(root, Gene_Expression_Subgroup, "Unclass", "GCB", "ABC").grid(row=32,column=1)
Label(root, text= "Gene Expression Subgroup:").grid(row=32,column=0)

Genetic_Subtype = StringVar(root)
Genetic_Subtype1 = OptionMenu(root, Genetic_Subtype,"EZB","BN2","N1","MCD", "Other").grid(row=33,column=1)
Label(root, text= "Genetic Subtype:").grid(row=33,column=0)

IPI_Group= StringVar(root)
IPI_Group1 = OptionMenu(root, IPI_Group,"Low", "Intermediate", "High").grid(row=34,column=1)
Label(root, text= "IPI Group: ").grid(row=34,column=0)

Treatment = StringVar(root)
Treatment1 = OptionMenu(root, Treatment,"Immunochemotherapy", "Ibrutinib monotherapy", "Unknown").grid(row=35,column=1)
Label(root, text= "Treatment: ").grid(row=35,column=0)

Biopsy_type = StringVar(root)
Biopsy_type1 = OptionMenu(root, Biopsy_type, "Pre-treatment", "Relapse").grid(row=36,column=1)
Label(root, text = "Biopsy Type:").grid(row=36,column=0)


cat_columns = [Gender, Gene_Expression_Subgroup, Genetic_Subtype, IPI_Group, Treatment,\
               Biopsy_type]


for col in numeric_coluns:
    globals()['label %s' % col] = Label(root, text=col + ":")
    globals()['label %s' % col].grid(row=row+1, column=0)
    globals()['button %s' % col] = Entry(root, width=30)
    globals()['button %s' % col].grid(row=row+1, column=1)
    row+=2


submit_button = Button(root, text = 'Submit', command = compute_results, bg = "#0099ff").grid(row=40, column=1)
root.mainloop()
