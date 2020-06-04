import pickle
import sklearn
import pandas as pd
import flask

app = flask.Flask(__name__, template_folder='templates')
with open('model/atrmodel.pkl', 'rb') as f:
    dtreemodel = pickle.load(f)

with open('model/atrmodelrandomforestclassifier.pkl', 'rb') as f:
    rfmodel = pickle.load(f)

with open('model/atrmodelsvm.pkl', 'rb') as f:
    svmmodel = pickle.load(f)

with open('model/atrmodellr.pkl', 'rb') as f:
    lrmodel = pickle.load(f)
    
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        data = pd.read_csv(flask.request.files.get('pickedfile'))
        shape=data.shape[0]
        data['YEAR_AT_COMPANY'].fillna((data['YEAR_AT_COMPANY'].mean()), inplace=True)
        data['NO_OF_DEPENDENT'] = data['NO_OF_DEPENDENT'].fillna(data['NO_OF_DEPENDENT'].mode()[0])
        data['CHILDREN'] = data['CHILDREN'].fillna(data['CHILDREN'].mode()[0])
        data.AGE_RANGE.replace(to_replace = {'LESS THAN 25 YEARS OLD': 0, '25 to 30 YEARS OLD': 1, '30 to 35 YEAR OLD': 2,'MORE THAN 35 YEARS OLD': 3}, inplace = True)
        data.GENDER.replace(to_replace = dict(MALE = 1, FEMALE = 0), inplace = True)
        data.GRADE.replace(to_replace = {'A+':0, 'A':1,'B+':2, 'B':3,'C':4, 'COMMANDO':5, 'STAFF':6}, inplace = True)
        data.MACHINIST_NONMACHINIST.replace(to_replace = {'MACHINIST':1, 'NON MACHINIST':0}, inplace = True)
        data.DEPARTMENT.replace(to_replace = {'SEWING':0,'FINISHING':1, 'CUTTING':2, 'FACTORY TRIM STORE':3,'HOUSE KEEPING':4,'HR & ADMIN':5,'DISPATCH':6}, inplace = True)

        data.SECTION.replace(to_replace = {'LINE C':0,'LINE A':1,'LINE B':2,'FINISHING':3,'SEWING':4,'CUTTING':5,'STORE':6,'HOUSE KEEPING':7,'NEEDLE STORE':8,'MATCHING':9,'CREACH':10,'DISPATCH':11,'SCANNER':12,'HR ADMIN':13,'TRAINING':14}, inplace = True)

        data.MARITAL_STATUS.replace(to_replace = {'MARRIED':1,'UNMARRIED':0}, inplace = True)
        data.LOCAL_MIG.replace(to_replace = {'LOCAL':0, 'MIGRANT':1, 'MIGRANT LOCAL':2, 'LOCAL/MIG':3}, inplace = True)
        data.LOCALITY.replace(to_replace = {'KALLABALU':0,'MADAPATTANA':1,'BOMMANAHALLI':2,'VINAYAKA NAGAR':3,'MAHANTHALINGAPURA':4,'BUKKASAGARA':5,'HARAGADDE':6,'BANNERGHATTA':7,'MARALAVADI':8,'VADER MANCHENAHALLI':9,'TATTEKERE':10,'HOSAROAD':11,'KRISHNA DODDI':12,'APC CIRCLE':13,'DEVASANDRA':14,'KONSANDRA':15,'JANGLE PALYA':16,'MANCHENAHALLI':17,'ANEKAL':18,'JIGANI':19,'KAGGALIPURA':20,'HEBBAGODI':21,'THIRUPALYA':22,'HOSUR':23,'HARAPANAHALLI':24,'INGALVADI':25,'KOPPAGATE':26,'BOMMASANDRA':27,'VADDARA PALYA':28,'YARANDAHALLI':29,'MARUTHI NAGAR':30,'YALLAMMANADODDI':31,'URAGANADODDI':32,' CHIKKENAHALLI':33,'HAREKADAKALU':34,'GIDDENAHALLI':35,'GOPAL LAYOUT':36,'KENGERI':37,'ATTIBELE':38,'NOSENUR':39,'SOPPALLI':40,'KANAKAPURA':41,'KONAPPANA AGRAHARA':42,'HENNAGARA':43,'MARATI BEEDI':44,'KALLANAKUPPE':45,'AADHUR':46,'KODICHIKKANHALLI':47,'BHARDRAPALYA':48,'RAJAPURA':49,'DASARAHALLI':50,'LINGAPUR':51,'KASABA HOBLI':52,'ANKODI':53,'KODIGEHALLI':54}, inplace = True)

        data.DIST_FROM_HOME.replace(to_replace = {'0-3 km':0, '3-6 km':1, '6-10 km':2, '10-15 km':3,'15-20 km':4, '20-25 km':5,'Above 25 km':6}, inplace=True)

        data.STATE.replace(to_replace = {'KARNATAKA':0, 'TAMIL NADU':1, 'ODISSA':2, 'BIHAR':3, 'UTTAR PRADESH':4, 'ANDRA PRADESH':5,'JHARKHAND':6, 'NEPAL':7, 'ASSAM':8, 'WEST BENGAL':9, 'MADHYA PRADESH':10}, inplace = True)

        data.HEALTH_ISSUES.replace(to_replace = {'HEALTH ISSUES':1, 'NO ISSUES':0}, inplace = True)
        data.ABS_GRADE.replace(to_replace = {'HIGH ABS':0, 'LOW ABS':1}, inplace = True)
        data.CHRONIC_NONCHRONIC.replace(to_replace = {'CHRONIC':1, 'NON CHRONIC':0}, inplace = True)
        data.EFFICIENY_RANGE.replace(to_replace = {'<40%':0,'40% TO 60%':1, '60% TO 80%':2,'80% TO 100%':3, '>100%':4}, inplace = True)
        data.NO_OF_DEPENDENT.replace(to_replace = {'5 AND ABOVE':5}, inplace = True)
        data.DESIGNATION.replace(to_replace = {'TAILOR':0, 'HELPER':1, 'KAJA BUTTON OPERATOR':2, 'IRONER':3, 'CHECKER':4,'SEWING ASSISTANT':5, 'BUTTON FIXING':6, 'TRIM&EXAM':7,'PACKERS/CARTONING':8,'FEEDING HELPER':9,'ALTRATION TAILOR':10, 'TAGING':11, 'BAGING':12,'MEASUREMENT CHECKER':13,'BUTTON MARKING':14, 'SPOT WASH':15, 'SUPERVISOR':16,'FINISHING ASSISTANT':17,'KANBAN':18,'CUTTER':19,'OPENING':20,'EDGE CUTTER':21,'NUMBERING':22,'RELAYER':23,'RE-CUTTER':24,'LAYER':25, 'PANEL CHECKER':26,'NEEDLE STORE':27,'CARE TAKER':28,'CUTTING ASSISTANT':29,'CREACH HELPER':30,'SCANNER':31,'DISPATCH ASSISTANT':32, 'MATCHING':33, 'PACKER':34, 'FINAL CHECKER':35,'HOUSE KEEPING':36,'FINISHING IRONER':37,'FOLDER':38}, inplace = True)

        data.INCREMENT_RANGE.replace(to_replace = {'NO INCREMENT':0,'LESS THAN 200':1,'BET 200-400':2, 'BET 400-600':3,'BET 600-800':4,'BET 800-1000':5,'BET 1000-1500':6,'BET 1500-2000':7,'MORE THAN 2000':8}, inplace = True)

        data.LOAN_RANGE.replace(to_replace = {'NO LOAN':0,'LESS THAN 2000':1,'BET 2000-4000':2, 'BET 4000-6000':3,'BET 6000-8000':4,'BET 8000-10000':5,'BET 10000-15000':6,'BET 15000-20000':7,'MORE THAN 20000':8}, inplace = True)

        data.REWARD.replace(to_replace = {'NO REWARD':0,'BEST OPERATOR':1,'MOST IMPROVED':2,'QUICK LEARNER':3}, inplace = True)
        data.AMBASSADOR.replace(to_replace = {'5S':1,'EMS':2,'VISION':3,'BEST OPERATOR':4,'CANTEEN COMMITTEE MEMBER':5,'RFT AMMABASSADOR':6,'WORKS COMMITTEE MEMBER':7}, inplace = True)

        EMPNO = data.EMPNO
        EMPNAME = data.NAME
        EMPST=data.STATUS
        data = data.drop(columns=['EMPNO','NAME','Unnamed: 0','STATUS'])
        data = sklearn.preprocessing.StandardScaler().fit(data).transform(data)
        x1=dtreemodel.predict(data)
        x2=rfmodel.predict(data)
        x3=svmmodel.predict(data)
        x4=lrmodel.predict(data)
        a0=[]
        a1=[]
        a2=[]
        a3=[]
        a4=[]
        for i in range(shape):
            x=[]
            identity = EMPNO[i]+': '+EMPNAME[i]
            x.append(identity)
            x.append(x1[i]+x2[i]+x3[i]+x4[i])
            if x[1]==0:
                a0.append(x[0])
            elif x[1]==1:
                a1.append(x[0])
            elif x[1]==2:
                a2.append(x[0])
            elif x[1]==3:
                a3.append(x[0])
            else:
                a4.append(x[0])
        #a=sorted(a, key = lambda x: x[1])
        str1='Zero Risk'
        str2='Low Risk'
        str3='Mild Risk'
        str4='High Risk'
        str5='Full Risk'        
        return flask.render_template('main.html', prediction0=a0, prediction1=a1, prediction2=a2, prediction3=a3, prediction4=a4, s1=str1, s2 =str2,s3=str3, s4=str4, s5=str5)    
        #return flask.render_template('main.html',prediction='{}'.format(count))
if __name__ == '__main__':
    app.run()
