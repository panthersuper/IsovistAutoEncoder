import numpy as np
import torch
import sys
sys.path.insert(0, '../')

from torch.autograd import Variable
import torch.nn as nn
from model.AutoEncoderResidualSeg import Net
import apiFunctions as API
import json
from trueskill import Rating, rate_1vs1

data = json.load(open('spacesurvey-pwz-export.json'))

AB_isovist = json.load(open('models/AB_isovist.json'))
BP_isovist = json.load(open('models/BP_isovist.json'))
EHB_isovist = json.load(open('models/EHB_isovist.json'))
VE_isovist = json.load(open('models/VE_isovist.json'))

start_from = '../trained/t92v96e56821'#'./alexnet64/Epoch28'
net = Net()
net.load_state_dict(torch.load(start_from, map_location={'cuda:0': 'cpu'}))

def man_dis(v1,v2):
    count = 0
    for vv1,vv2 in zip(v1,v2):
        count += abs(float(vv1)-float(vv2))
    
    return count


output_pairs = {}

# individual vector list, each vector have values acumulated from the pairs. vector is used as key
output_solo = []

v_list = []
v_indexs = {}
v1_indexs = []
v2_indexs = []
dis_lst = []

count = 0
for key in data:
    if(key != "debug"):
        thisd = data[key]
        s1_index = thisd['scene1_index']
        s1_model = thisd['scene1_model']
        s2_index = thisd['scene2_index']
        s2_model = thisd['scene2_model']

        v1 = globals()[s1_model+"_isovist"][int(s1_index)]
        v2 = globals()[s2_model+"_isovist"][int(s2_index)]

        v1 = np.array(v1)/10000
        v2 = np.array(v2)/10000

        # encoded version of v1 and v2
        v1 = API.encode(v1,net).tolist()
        v2 = API.encode(v2,net).tolist()

        # add vs to v_list
        v1_dumps = json.dumps(v1)
        v2_dumps = json.dumps(v2)
        
        if(v1_dumps not in v_list):
            v_list.append(v1_dumps)
            v_indexs[v1_dumps] = {
                'model':s1_model,
                'index':s1_index
            }
        if(v2_dumps not in v_list):
            v_list.append(v2_dumps)
            v_indexs[v2_dumps] = {
                'model':s2_model,
                'index':s2_index
            }

        output_pairs[key] = {}
        output_pairs[key]["age"] = thisd["age"]
        output_pairs[key]["arch_background"] = thisd["arch_background"]
        output_pairs[key]["gender"] = thisd["gender"]
        output_pairs[key]["interest"] = thisd["interest"]
        output_pairs[key]["public"] = thisd["public"]
        output_pairs[key]["spacious"] = thisd["spacious"]
        output_pairs[key]["v1"] = v1_dumps
        output_pairs[key]["v2"] = v2_dumps

        # scene 1 is selected
        if(thisd["interest"] == "0" or thisd["public"] == "0" or thisd["spacious"] == "0"):
            solo1 = {}
            solo1["v"] = v1_dumps
            solo1["interest"] = 1 if thisd["interest"]=="0" else 0
            solo1["public"] = 1 if thisd["public"]=="0" else 0
            solo1["spacious"] = 1 if thisd["spacious"]=="0" else 0
            output_solo.append(solo1)
        
        if(thisd["interest"] == "1" or thisd["public"] == "1" or thisd["spacious"] == "1"):
            solo2 = {}
            solo2["v"] = v2_dumps
            solo2["interest"] = 1 if thisd["interest"]=="1" else 0
            solo2["public"] = 1 if thisd["public"]=="1" else 0
            solo2["spacious"] = 1 if thisd["spacious"]=="1" else 0
            output_solo.append(solo2)


        dis_lst.append(man_dis(v1,v2))
        count +=1
        print(count)


with open('processed_pairs.json', 'w') as outfile:
    json.dump(output_pairs, outfile)

with open('processed_solo.json', 'w') as outfile:
    json.dump(output_solo, outfile)

print("range of distance ",min(dis_lst),max(dis_lst))

print("vs_total ",count*2)
print("vs ",len(v_list))
print("vs_solo ",len(output_solo))

# create trueSkill rating list for each vs, use vector string as key
interast_list = {}
public_list = {}
spacious_list = {}

rating_list = {}

for v in v_list:
    rating_list[v] = {}

    rating_list[v]["interest"] = Rating()
    rating_list[v]["public"] = Rating()
    rating_list[v]["spacious"] = Rating()

    # interast_list[v] = Rating()
    # public_list[v] = Rating()
    # spacious_list[v] = Rating()

count2 = 0
count_pairs = 0
for key in output_pairs:
    count2 +=1
    pair = output_pairs[key]
    v1 = pair["v1"]
    v2 = pair["v2"]

    v1_json = json.loads(v1)
    v2_json = json.loads(v2)

    # v1 won interest
    if pair["interest"] == "0":
        rating_list[v1]["interest"],rating_list[v2]["interest"] = rate_1vs1(rating_list[v1]["interest"], rating_list[v2]["interest"])
    else:
        rating_list[v2]["interest"],rating_list[v1]["interest"] = rate_1vs1(rating_list[v2]["interest"], rating_list[v1]["interest"])
    if pair["public"] == "0":
        rating_list[v1]["public"],rating_list[v2]["public"] = rate_1vs1(rating_list[v1]["public"], rating_list[v2]["public"])
    else:
        rating_list[v2]["public"],rating_list[v1]["public"] = rate_1vs1(rating_list[v2]["public"], rating_list[v1]["public"])
    if pair["spacious"] == "0":
        rating_list[v1]["spacious"],rating_list[v2]["spacious"] = rate_1vs1(rating_list[v1]["spacious"], rating_list[v2]["spacious"])
    else:
        rating_list[v2]["spacious"],rating_list[v1]["spacious"] = rate_1vs1(rating_list[v2]["spacious"], rating_list[v1]["spacious"])

    count_pairs +=1

    print(count2)
    for key2 in output_pairs:
        if key != key2:
            pair2 = output_pairs[key2]
            v11 = pair2["v1"]
            v22 = pair2["v2"]
            v11_json = json.loads(v11)
            v22_json = json.loads(v22)

            # v1 is almost v11, threshold can be changed
            if(man_dis(v1_json,v11_json)<5):
                # v1 won interest
                if pair["interest"] == "0":
                    rating_list[v11]["interest"],rating_list[v2]["interest"] = rate_1vs1(rating_list[v11]["interest"], rating_list[v2]["interest"])
                else:
                    rating_list[v2]["interest"],rating_list[v11]["interest"] = rate_1vs1(rating_list[v2]["interest"], rating_list[v11]["interest"])
                if pair["public"] == "0":
                    rating_list[v11]["public"],rating_list[v2]["public"] = rate_1vs1(rating_list[v11]["public"], rating_list[v2]["public"])
                else:
                    rating_list[v2]["public"],rating_list[v11]["public"] = rate_1vs1(rating_list[v2]["public"], rating_list[v11]["public"])
                if pair["spacious"] == "0":
                    rating_list[v11]["spacious"],rating_list[v2]["spacious"] = rate_1vs1(rating_list[v11]["spacious"], rating_list[v2]["spacious"])
                else:
                    rating_list[v2]["spacious"],rating_list[v11]["spacious"] = rate_1vs1(rating_list[v2]["spacious"], rating_list[v11]["spacious"])
                count_pairs +=1

                # v11 won interest
                if pair2["interest"] == "0":
                    rating_list[v1]["interest"],rating_list[v22]["interest"] = rate_1vs1(rating_list[v1]["interest"], rating_list[v22]["interest"])
                else:
                    rating_list[v22]["interest"],rating_list[v1]["interest"] = rate_1vs1(rating_list[v22]["interest"], rating_list[v1]["interest"])
                if pair2["public"] == "0":
                    rating_list[v1]["public"],rating_list[v22]["public"] = rate_1vs1(rating_list[v1]["public"], rating_list[v22]["public"])
                else:
                    rating_list[v22]["public"],rating_list[v1]["public"] = rate_1vs1(rating_list[v22]["public"], rating_list[v1]["public"])
                if pair2["spacious"] == "0":
                    rating_list[v1]["spacious"],rating_list[v22]["spacious"] = rate_1vs1(rating_list[v1]["spacious"], rating_list[v22]["spacious"])
                else:
                    rating_list[v22]["spacious"],rating_list[v1]["spacious"] = rate_1vs1(rating_list[v22]["spacious"], rating_list[v1]["spacious"])
                count_pairs +=1

            # v1 is almost v22, threshold can be changed
            if(man_dis(v1_json,v22_json)<5):
                # v1 won interest
                if pair["interest"] == "0":
                    rating_list[v22]["interest"],rating_list[v2]["interest"] = rate_1vs1(rating_list[v22]["interest"], rating_list[v2]["interest"])
                else:
                    rating_list[v2]["interest"],rating_list[v22]["interest"] = rate_1vs1(rating_list[v2]["interest"], rating_list[v22]["interest"])
                if pair["public"] == "0":
                    rating_list[v22]["public"],rating_list[v2]["public"] = rate_1vs1(rating_list[v22]["public"], rating_list[v2]["public"])
                else:
                    rating_list[v2]["public"],rating_list[v22]["public"] = rate_1vs1(rating_list[v2]["public"], rating_list[v22]["public"])
                if pair["spacious"] == "0":
                    rating_list[v22]["spacious"],rating_list[v2]["spacious"] = rate_1vs1(rating_list[v22]["spacious"], rating_list[v2]["spacious"])
                else:
                    rating_list[v2]["spacious"],rating_list[v22]["spacious"] = rate_1vs1(rating_list[v2]["spacious"], rating_list[v22]["spacious"])
                count_pairs +=1

                # v22 won interest
                if pair2["interest"] == "1":
                    rating_list[v1]["interest"],rating_list[v11]["interest"] = rate_1vs1(rating_list[v1]["interest"], rating_list[v11]["interest"])
                else:
                    rating_list[v11]["interest"],rating_list[v1]["interest"] = rate_1vs1(rating_list[v11]["interest"], rating_list[v1]["interest"])
                if pair2["public"] == "1":
                    rating_list[v1]["public"],rating_list[v11]["public"] = rate_1vs1(rating_list[v1]["public"], rating_list[v11]["public"])
                else:
                    rating_list[v11]["public"],rating_list[v1]["public"] = rate_1vs1(rating_list[v11]["public"], rating_list[v1]["public"])
                if pair2["spacious"] == "1":
                    rating_list[v1]["spacious"],rating_list[v11]["spacious"] = rate_1vs1(rating_list[v1]["spacious"], rating_list[v11]["spacious"])
                else:
                    rating_list[v11]["spacious"],rating_list[v1]["spacious"] = rate_1vs1(rating_list[v11]["spacious"], rating_list[v1]["spacious"])
                count_pairs +=1

            # v2 is almost v11
            if(man_dis(v2_json,v11_json)<5):
                # v2 won interest
                if pair["interest"] == "1":
                    rating_list[v11]["interest"],rating_list[v1]["interest"] = rate_1vs1(rating_list[v11]["interest"], rating_list[v1]["interest"])
                else:
                    rating_list[v1]["interest"],rating_list[v11]["interest"] = rate_1vs1(rating_list[v1]["interest"], rating_list[v11]["interest"])
                if pair["public"] == "1":
                    rating_list[v11]["public"],rating_list[v1]["public"] = rate_1vs1(rating_list[v11]["public"], rating_list[v1]["public"])
                else:
                    rating_list[v1]["public"],rating_list[v11]["public"] = rate_1vs1(rating_list[v1]["public"], rating_list[v11]["public"])
                if pair["spacious"] == "1":
                    rating_list[v11]["spacious"],rating_list[v1]["spacious"] = rate_1vs1(rating_list[v11]["spacious"], rating_list[v1]["spacious"])
                else:
                    rating_list[v1]["spacious"],rating_list[v11]["spacious"] = rate_1vs1(rating_list[v1]["spacious"], rating_list[v11]["spacious"])
                count_pairs +=1

                # v11 won interest
                if pair2["interest"] == "0":
                    rating_list[v2]["interest"],rating_list[v22]["interest"] = rate_1vs1(rating_list[v2]["interest"], rating_list[v22]["interest"])
                else:
                    rating_list[v22]["interest"],rating_list[v2]["interest"] = rate_1vs1(rating_list[v22]["interest"], rating_list[v2]["interest"])
                if pair2["public"] == "0":
                    rating_list[v2]["public"],rating_list[v22]["public"] = rate_1vs1(rating_list[v2]["public"], rating_list[v22]["public"])
                else:
                    rating_list[v22]["public"],rating_list[v2]["public"] = rate_1vs1(rating_list[v22]["public"], rating_list[v2]["public"])
                if pair2["spacious"] == "0":
                    rating_list[v2]["spacious"],rating_list[v22]["spacious"] = rate_1vs1(rating_list[v2]["spacious"], rating_list[v22]["spacious"])
                else:
                    rating_list[v22]["spacious"],rating_list[v2]["spacious"] = rate_1vs1(rating_list[v22]["spacious"], rating_list[v2]["spacious"])
                count_pairs +=1

            # v2 is almost v22
            if(man_dis(v2_json,v22_json)<5):
                # v2 won interest
                if pair["interest"] == "1":
                    rating_list[v22]["interest"],rating_list[v1]["interest"] = rate_1vs1(rating_list[v22]["interest"], rating_list[v1]["interest"])
                else:
                    rating_list[v1]["interest"],rating_list[v22]["interest"] = rate_1vs1(rating_list[v1]["interest"], rating_list[v22]["interest"])
                if pair["public"] == "1":
                    rating_list[v22]["public"],rating_list[v1]["public"] = rate_1vs1(rating_list[v22]["public"], rating_list[v1]["public"])
                else:
                    rating_list[v1]["public"],rating_list[v22]["public"] = rate_1vs1(rating_list[v1]["public"], rating_list[v22]["public"])
                if pair["spacious"] == "1":
                    rating_list[v22]["spacious"],rating_list[v1]["spacious"] = rate_1vs1(rating_list[v22]["spacious"], rating_list[v1]["spacious"])
                else:
                    rating_list[v1]["spacious"],rating_list[v22]["spacious"] = rate_1vs1(rating_list[v1]["spacious"], rating_list[v22]["spacious"])
                count_pairs +=1

                # v22 won interest
                if pair2["interest"] == "1":
                    rating_list[v2]["interest"],rating_list[v11]["interest"] = rate_1vs1(rating_list[v2]["interest"], rating_list[v11]["interest"])
                else:
                    rating_list[v11]["interest"],rating_list[v2]["interest"] = rate_1vs1(rating_list[v11]["interest"], rating_list[v2]["interest"])
                if pair2["public"] == "1":
                    rating_list[v2]["public"],rating_list[v11]["public"] = rate_1vs1(rating_list[v2]["public"], rating_list[v11]["public"])
                else:
                    rating_list[v11]["public"],rating_list[v2]["public"] = rate_1vs1(rating_list[v11]["public"], rating_list[v2]["public"])
                if pair2["spacious"] == "1":
                    rating_list[v2]["spacious"],rating_list[v11]["spacious"] = rate_1vs1(rating_list[v2]["spacious"], rating_list[v11]["spacious"])
                else:
                    rating_list[v11]["spacious"],rating_list[v2]["spacious"] = rate_1vs1(rating_list[v11]["spacious"], rating_list[v2]["spacious"])
                count_pairs +=1


interast_list = [rating_list[key]["interest"].mu for key in rating_list]
public_list = [rating_list[key]["public"].mu for key in rating_list]
spacious_list = [rating_list[key]["spacious"].mu for key in rating_list]


for key in rating_list:
    rating_list[key]["interest"] = (rating_list[key]["interest"].mu - min(interast_list))/(max(interast_list)-min(interast_list))
    rating_list[key]["public"] = (rating_list[key]["public"].mu - min(public_list))/(max(public_list)-min(public_list))
    rating_list[key]["spacious"] = (rating_list[key]["spacious"].mu - min(spacious_list))/(max(spacious_list)-min(spacious_list))
    rating_list[key]["indexs"] = v_indexs[key]


# print(interast_list)
print("original pairs:",count)
print("final pairs:",count_pairs)

with open('normalized_ratings.json', 'w') as outfile:
    json.dump(rating_list, outfile)





