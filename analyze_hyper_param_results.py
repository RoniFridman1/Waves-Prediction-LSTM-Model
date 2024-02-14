import glob
import os, pandas as pd
import re

results_path = r"C:\Users\soldier109\PycharmProjects\CAMERI_Work\waves_height_prediction\images_from_db77"
results_list = []
loss_list = []
for file in glob.glob(os.path.join(results_path,"*.txt")):
    idx = os.path.basename(file)[:-4].split("_")[1]
    curr_result = open(file,"r").readlines()
    curr_result = [line[:-1] for line in curr_result if line !="\n"]
    # train_loss = curr_result[-2]
    test_loss = curr_result[-1].split(":")[1].split(",")
    test_loss = [float(re.sub("\s+|\[|\]]","",i)) for i in test_loss]
    curr_result = [curr_result[9],curr_result[10],curr_result[15],curr_result[16],curr_result[18],curr_result[20].split("  #")[0]]
    curr_result = [float(l.split("=")[1]) for l in curr_result]
    results_list.append([idx]+curr_result)
    loss_list.append([idx]+test_loss)
    # order : years10, ratio11, hidden_units16, lr17, batch_size19, seq_len_div21

loss_columns = ['idx']+[str(i) for i in range(19)]
results_columns = ['idx']+["YEAR_TO_USE","TRAIN_TEST_RATIO","NUM_HIDDEN_UNITS","LEARNING_RATE","BATCH_SIZE","SEQ_DIVIDER"]
loss_df = pd.DataFrame(loss_list,columns=loss_columns)
results_df = pd.DataFrame(results_list, columns=results_columns)
loss_df.set_index("idx",inplace=True)
results_df.set_index("idx",inplace=True)
loss_df.to_csv(".\\loss_df.csv")
results_df.to_csv(".\\results_df.csv")
print(loss_df)
print(results_df)