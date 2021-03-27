from flask import Flask, render_template, url_for, request

app = Flask(__name__)

# Main page
@app.route('/', methods=['GET','POST'])
def index():
    title = "HR Analytics"
    gender_cat = ['Female', 'Male', 'Other']
    city_cat = ['city_103', 'city_40', 'city_21', 'city_115', 'city_162',
       'city_176', 'city_160', 'city_46', 'city_61', 'city_114',
       'city_13', 'city_159', 'city_102', 'city_67', 'city_100',
       'city_16', 'city_71', 'city_104', 'city_64', 'city_101', 'city_83',
       'city_105', 'city_73', 'city_75', 'city_41', 'city_11', 'city_93',
       'city_90', 'city_36', 'city_20', 'city_57', 'city_152', 'city_19',
       'city_65', 'city_74', 'city_173', 'city_136', 'city_98', 'city_97',
       'city_50', 'city_138', 'city_82', 'city_157', 'city_89',
       'city_150', 'city_70', 'city_175', 'city_94', 'city_28', 'city_59',
       'city_165', 'city_145', 'city_142', 'city_26', 'city_12',
       'city_37', 'city_43', 'city_116', 'city_23', 'city_99', 'city_149',
       'city_10', 'city_45', 'city_80', 'city_128', 'city_158',
       'city_123', 'city_7', 'city_72', 'city_106', 'city_143', 'city_78',
       'city_109', 'city_24', 'city_134', 'city_48', 'city_144',
       'city_91', 'city_146', 'city_133', 'city_126', 'city_118',
       'city_9', 'city_167', 'city_27', 'city_84', 'city_54', 'city_39',
       'city_79', 'city_76', 'city_77', 'city_81', 'city_131', 'city_44',
       'city_117', 'city_155', 'city_33', 'city_141', 'city_127',
       'city_62', 'city_53', 'city_25', 'city_2', 'city_69', 'city_120',
       'city_111', 'city_30', 'city_1', 'city_140', 'city_179', 'city_55',
       'city_14', 'city_42', 'city_107', 'city_18', 'city_139',
       'city_180', 'city_166', 'city_121', 'city_129', 'city_8',
       'city_31', 'city_171']
    city_cat = sorted(list(city_cat))

    relevent_cat = ['Has relevent experience', 'No relevent experience']
    enrolled_cat = ['Full time course', 'Part time course', 'no_enrollment']
    education_cat = ['Graduate', 'Masters', 'Phd']
    major_cat = ['STEM', 'Business Degree', 'Arts', 'Humanities', 'No Major', 'Other']
    company_size_cat = ['<10', '10-49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+']
    company_type_cat = ['Pvt Ltd', 'Funded Startup', 'Early Stage Startup', 'Public Sector', 'NGO', 'Other']
    last_new_job_cat = ['never', '1', '2', '3', '4', '>4']

    if request.method == 'GET':
        label = 'Oops! Please fill employee criterion and submit'
        return render_template('index.html', title=title, gender_cat=gender_cat, city_cat=city_cat,
                                relevent_cat=relevent_cat, enrolled_cat=enrolled_cat, education_cat=education_cat,
                                major_cat=major_cat, company_size_cat=company_size_cat, company_type_cat=company_type_cat,
                                last_new_job_cat=last_new_job_cat)
    else:
        import joblib
        # from sklearn.externals.joblib import dump, load
        from tensorflow.keras.models import load_model
        import numpy as np
        import pandas as pd
        
        gender = request.form['gender']
        city_development_index = request.form['city_development_index']
        relevent_experience = request.form['relevent_experience']
        enrolled_university = request.form['enrolled_university']
        education_level = request.form['education_level']
        major_discipline = request.form['major_discipline']
        company_size = request.form['company_size']
        company_type = request.form['company_type']
        last_new_job = request.form['last_new_job']
        experience = request.form['experience']
        training_hours = request.form['training_hours']

        data_test = {'city_development_index': float(city_development_index), 'relevent_experience': relevent_experience, 
                    'education_level':education_level, 'experience': int(experience), 'company_size': company_size, 
                    'last_new_job': last_new_job, 'training_hours': int(training_hours), 'gender': gender, 
                    'enrolled': enrolled_university, 'major': major_discipline, 'company_type': company_type}

        data_prep = preprocessing_data(data_test)
        df_test = pd.DataFrame(columns=list(data_prep.keys()))
        df_test.loc[0] = data_prep.values()

        num = ['city_development_index', 'training_hours']
        df_test_num = df_test.loc[:, num]
        df_test_cat = df_test.drop(num, axis=1)
        df_test_cat.reset_index(drop=True, inplace=True)

        scaler = joblib.load('model/std_scaler.bin')
        df_test_num_std = scaler.transform(df_test_num)
        df_test = pd.concat([df_test_cat, pd.DataFrame(df_test_num_std)], axis=1)

        loaded_model = load_model("model/model_8598.h5")
        score = round(loaded_model.predict(df_test)[0][0])

        if score == 0:
            label = 'Not looking for a job change'
        elif score == 1:
            label = 'Loking for a job change'

        return render_template('index.html', title=title, gender_cat=gender_cat, city_cat=city_cat,
                                relevent_cat=relevent_cat, enrolled_cat=enrolled_cat, education_cat=education_cat,
                                major_cat=major_cat, company_size_cat=company_size_cat, company_type_cat=company_type_cat,
                                last_new_job_cat=last_new_job_cat, label=label)

def preprocessing_data(dt):
    data_prep = dt.copy()

    def education_category(x):
        category = 0
        if x == 'Graduate':
            category = 0
        elif x == 'Masters':
            category = 1
        elif x == 'Phd':
            category = 2
        return category
  
    def experience_category(x):
        category = 0
        if x <= 20:
            category = 0
        else:
            category = 1
        return category
    
    def company_size_category(x):
        category = 0
        if x in ['<10', '10/49', '50-99', '100-500', '500-900']:
            category = 0
        elif x in ['1000-4999', '5000-9999', '10000+']:
            category = 1
        return category
    
    def last_job_category(x):
        category = 0
        if x in ['never', '1']:
            category = 0
        elif x in ['2', '3', '4', '>4', '5']:
            category = 1
        return category
  
    data_prep['relevent_experience'] = 1 if data_prep['relevent_experience'] == 'Has relevent experience' else 0
    data_prep['education_level'] = education_category(data_prep['education_level'])
    # data_prep['experience'] =  21 if data_prep['experience'] == '>20' else ( 0 if data_prep['experience'] == '<1' else int(data_prep['experience']))
    data_prep['experience'] = experience_category(int(data_prep['experience']))
    data_prep['company_size'] = company_size_category(data_prep['company_size'])
    data_prep['last_new_job'] = last_job_category(data_prep['last_new_job'])

    gender = data_prep['gender']
    gender_cat = ['Female', 'Male', 'Other']
    del data_prep['gender']
    for cat in gender_cat:
        if cat == gender:
            data_prep['gender_' + cat] = 1
        else:
            data_prep['gender_' + cat] = 0

    enrolled = data_prep['enrolled']
    enrolled_cat = ['Full time course', 'Part time course', 'no_enrollment']
    del data_prep['enrolled']
    for cat in enrolled_cat:
        if cat == enrolled:
            data_prep['enroll_' + cat] = 1
        else:
            data_prep['enroll_' + cat] = 0
    
    major = data_prep['major']
    major_cat = ['Arts','Business Degree', 'Humanities', 'No Major', 'Other', 'STEM']
    del data_prep['major']
    for cat in major_cat:
        if cat == major:
            data_prep['major_' + cat] = 1
        else:
            data_prep['major_' + cat] = 0

    company_type = data_prep['company_type']
    company_type_cat = ['Early Stage Startup', 'Funded Startup', 'NGO', 'Other', 'Public Sector', 'Pvt Ltd']
    del data_prep['company_type']
    for cat in company_type_cat:
        if cat == company_type:
            data_prep['company_type_' + cat] = 1
        else:
            data_prep['company_type_' + cat] = 0
    
    return data_prep

if __name__ == '__main__':
    app.run()