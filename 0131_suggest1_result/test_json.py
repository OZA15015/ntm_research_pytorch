import json

with open("copy-task-1000-batch-50000.json") as f:
    jsn = json.load(f)

print(type(jsn["loss"]))

for jsn_key in jsn:
    print(jsn_key)
cost_sum = 0
loss_sum = 0
seq_sum = 0
flag = 0
count = 0
for i in range(len(jsn["loss"])):

    '''
    print("loss: ", end = "")
    print(jsn["loss"][i], end = "")
    print(", cost: ", end = "")
    print(jsn["cost"][i], end = "")
    print(", seq_lengths: ", end = "")
    print(jsn["seq_lengths"][i])
    '''
    if(i >= 49000):
	    cost_sum += jsn["cost"][i]
	    loss_sum += jsn["loss"][i]
	    seq_sum +=  jsn["seq_lengths"][i]
	    count += 1
print(len(jsn["loss"]))
print(len(jsn["seq_lengths"]))
print(len(jsn["cost"]))

print("seq_ave: ", end = "")
#print(seq_sum / len(jsn["loss"]), end = "")
print(seq_sum / 1000, end = "")
print(", cost_ave: ", end = "")
#print(cost_sum / len(jsn["loss"]), end = "")
print(cost_sum / 1000, end = "")
print(", loss_ave: ", end = "")
#print(loss_sum / len(jsn["loss"]))
print(loss_sum / 1000)
print(count)
