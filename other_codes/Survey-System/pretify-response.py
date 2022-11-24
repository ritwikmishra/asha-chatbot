
import pandas as pd
import os

data = pd.read_csv("final-Ques.csv")

def nameopt(rval):
    if rval == 0:
        return ""
    elif rval == 1:
        return "Totally relevant"
    elif rval == 2:
        return "Somewhat relevant"
    elif rval == 3:
        return "Irrelevant"

def main():
    finaldir = "FinalResponse"
    for fname in os.listdir("response"):
        if fname != "empty.csv":
            ans = pd.read_csv("response/"+fname)
            ans.Ans = ans.Ans.apply(nameopt)
            ans_arr = list(ans.Ans.values)

            pred1 = ans_arr[0::3]
            pred2 = ans_arr[1::3]
            pred3 = ans_arr[2::3]

            data["EXP_pred_1"] = pred1
            data["EXP_pred_2"] = pred2
            data["EXP_pred_3"] = pred3

            data.to_csv(finaldir+"/"+fname,index = False)

if __name__ == "__main__":
    main()
    print("Done!")