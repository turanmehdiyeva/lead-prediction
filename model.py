#Vectorizing variables
cat = list(data_mi.iloc[0:9,:].index)
train_dict = train_X[cat+numerical].to_dict(orient='rows')
train_dict[0]

from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)
len(X_train[0])

dv.get_feature_names()

#Logistic regression
val_dict = val_X[cat+numerical].to_dict(orient='rows')
X_val = dv.transform(val_dict)
len(X_val[0])

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', random_state=1)
model.fit(X_train, train_y)

preds = model.predict_proba(X_val)[:,1]

pred = (preds>0.5).astype(int).reshape(-1,1)

#accuracy
(pred==val_y).mean()

#Further evaluation
from sklearn.metrics import accuracy_score
thresholds = np.linspace(0,1,11)

for i in thresholds:
    churn = preds>=i
    acc = accuracy_score(val_y, churn)
    print(f'{round(i,1)} {round(acc,3)}')
    
thresholds = np.linspace(0,1,21)
accs = []

for i in thresholds:
    churn = preds>=i
    acc = accuracy_score(val_y, churn)
    accs.append(acc)
    
plt.plot(thresholds, accs)
plt.xlabel('Thresholds')
plt.ylabel('Accuracy')
plt.title('Threshold vs Accuracy')
plt.show()

#Confusion Table
t = 0.4

predict_convert = (preds>=t).reshape(-1,1)
predict_no_convert = (preds<t).reshape(-1,1)

actual_convert = (val_y==1)
actual_no_convert = (val_y==0)

true_positive = (predict_convert & actual_convert).sum()
true_negative = (predict_no_convert & actual_no_convert).sum()

false_positive = (predict_convert & actual_no_convert).sum()
false_negative = (predict_no_convert & actual_convert).sum()

confussion_table = np.array([[true_negative, false_positive],[false_negative,true_positive]])
confussion_table

confussion_table/confussion_table.sum()

P = true_positive/(true_positive+false_positive)
print(f'Percent of correct predictions among leads predicted as converted is {round(P,2)*100}%')
R = true_positive/(true_positive+false_negative)
print(f'Percent of correct predictions among actual converteds is {round(R,2)*100}%')

