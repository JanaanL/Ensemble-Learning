
# coding: utf-8

# In[ ]:


# A python library for learning ensembles of decision trees:  AdaBoost, Bagging and Random Forest
# Created by Janaan Lake February 2019 for CS_5350
import numpy as np
import matplotlib as plt


# In[190]:


class Node(object):
    """
    A class for a tree node
    This class is specifically for a decision-type tree
    """
    
    def __init__(self, name='root', attribute=None, action=None, parent=None, branches=None):
        self.name = name
        self.attribute = attribute
        self.action = action
        self.parent = parent
        self.branches = {}
    
    def __repr__(self):
        return self.name
    
    def add_branch(self, value, node=None):
        self.branches[value] = node
        
    def get_branch(self, value):
        return self.branches[value]
    
    def print_tree(self):
        print("Type = " + self.name)
        if self.attribute is not None:
            print("Attribute = " + self.attribute)
        if self.parent is not None:
            print("Parent = " + self.parent)
        if self.action is not None:
            print("Action = " + self.action)
        if len(self.branches) > 0:
            for branch in self.branches:
                print("\n")
                print("Value = " + branch)
                self.branches[branch].print_tree()


# In[637]:


def majority_label(S, attribute="label", replace=False, weighted=False):
    """
    Determines the majority label of a given attribute.
    
    Input: 
    S:  A list of dictionaries with key-value pairs represented as strings.
    attribute: The attribute that will be searched 
    replace:  If replace is True, the value "unknown" will not be counted in determing the majority label
    
    Returns:  a string representing the most common value in the list for the key attribute 
    """
    counts = {}
    
    for s in S:
        if weighted:
            weight = s["weight"]
        else:
            weight = 1.0
            
        value = s[attribute]
        if value in counts:
            if value != "unknown":
                counts[value] += 1 * weight
            else:
                if not replace:
                    counts[value] += 1 * weight  
        else:
            if value != "unknown":
                counts[value] = 1 * weight
            else:
                if not replace:
                    counts[value] = 1 * weight 
    
    return max(counts, key=counts.get)


# In[629]:


def read_file(path, label):
    """
    Reads and processes a csv file for use in a decision tree algorithm.
    
    Input: 
    -path: a string representing the file path of the csv file to be opened.  
    -label: represents the type of examples to be processed.  One of two possible values: "bank","tennis"
    

    Returns:  
    -S: a list of dictionaries.  Each dictionary represents a single instance (or one line in the data file)
        with key-value pairs representing attributes and values.      
    """    
        
    S = []
    with open(path, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
                
            if label == "bank":
                    
                if len(terms) == 17:
                    example = {}
                    example["age"] = terms[0]
                    example["job"] = terms[1]
                    example["marital"] = terms[2]
                    example["education"] = terms[3]
                    example["default"] = terms[4]
                    example["balance"] = terms[5]
                    example["housing"] = terms[6]
                    example["loan"] = terms[7]
                    example["contact"] = terms[8]
                    example["day"] = terms[9]
                    example["month"] = terms[10]
                    example["duration"] = terms[11]
                    example["campaign"] = terms[12]
                    example["pdays"] = terms[13]
                    example["previous"] = terms[14]
                    example["poutcome"] = terms[15]
                    example["label"] = terms[16]
                    S.append(example)
                    
            if label == "tennis":
                if len(terms) == 5:
                    example = {}
                    example["outlook"] = terms[0]
                    example["temp"] = terms[1]
                    example["humidity"] = terms[2]
                    example["wind"] = terms[3]
                    example["label"] = terms[4]
                    S.append(example)
                    
    return S
            


# In[647]:


def process_bank_data(S, mode, train_medians=None, replace=False, maj_values=None):
    """
    A function that processes the bank data.  First, it turns the numerical range into binary data.  
    The binary values ("high", "low") are based on a threshold. The threshold is the median value in the set S.
    Also, it can replace "unknown" attribute values with the majority value for that attribute.
        
    Inputs:
    -S: Set of data instances from the bank data set. 
    -mode: One of two types: "train" or "test".  If "train", medians will be calculated from data.
        If "test", the medians will be provided in the function call.
    -train_medians: dictionary with key-value pairs being the attribute and its associated numerical median.
    -replace:  If False, "unknown" attribute values are considered a value.  Otherwise if True, "unknown" 
               attribute values are replaced with the majority value of that attribute.
    -maj_labels:  a dictionary with key-value pairs being the attribute and its majority label.
    
    Returns:
    -medians:  A dictionary with key-value pairs.  The key is the attribute represented as a string and 
        the value is the corresponding numerical median.
    -majority: A dictionary with key-value pairs .  The key is the attribute represented as a string and 
        the value is the corresponding majority element for that attribute
    """

    from statistics import median
    
    #Calculate the median if training mode
    if mode == "train":
        age = []
        balance = []
        day = []
        duration = []
        campaign = []
        pdays = []
        previous = []
    
        for s in S:
            age.append(int(s["age"]))
            balance.append(int(s["balance"]))
            day.append(int(s["day"]))
            duration.append(int(s["duration"]))
            campaign.append(int(s["campaign"]))
            pdays.append(int(s["pdays"]))
            previous.append(int(s["previous"]))
        
        age.sort()
        balance.sort()
        day.sort()
        duration.sort()
        campaign.sort()
        pdays.sort()
        previous.sort()
        
        med_age = median(age)
        med_balance = median(balance)
        med_day = median(day)
        med_duration = median(duration)
        med_campaign = median(campaign)
        med_pdays = median(pdays)
        med_previous = median(previous)
            
        if replace:
            maj_job = majority_label(S,"job", "True")
            maj_education = majority_label(S, "education", "True")
            maj_contact = majority_label(S, "contact", "True")
            maj_poutcome = majority_label(S, "poutcome", "True")
    
    else:
        med_age = train_medians["age"]
        med_balance = train_medians["balance"]
        med_day = train_medians["day"]
        med_duration = train_medians["duration"]
        med_campaign = train_medians["campaign"]
        med_pdays = train_medians["pdays"]
        med_previous = train_medians["previous"]
        
        if replace:
            maj_job = maj_values["job"]
            maj_education = maj_values["education"]
            maj_contact = maj_values["contact"]
            maj_poutcome = maj_values["poutcome"]


    for s in S:
        s["age"] = "high" if float(s["age"]) > med_age else "low"
        s["balance"] = "high" if float(s["balance"]) > med_balance else "low"
        s["day"] = "high" if float(s["day"]) > med_day else "low"
        s["duration"] = "high" if float(s["duration"]) > med_duration else "low"
        s["campaign"] = "high" if float(s["campaign"]) > med_campaign else "low"
        s["pdays"] = "high" if float(s["pdays"]) > med_pdays else "low"
        s["previous"] = "high" if float(s["previous"]) > med_previous else "low"
        
        if replace:
            if s["job"] == "unknown":
                s["job"] = maj_job 
            if s["education"] == "unknown":
                s["education"] = maj_education
            if s["contact"] == "unknown":
                s["contact"] = maj_contact 
            if s["poutcome"] == "unknown":
                s["poutcome"] = maj_poutcome

        
    medians = {
        "age": med_age,
        "balance": med_balance,
        "day": med_day,
        "duration": med_duration,
        "campaign": med_campaign,
        "pdays": med_pdays,
        "previous": med_previous
    }
    
    if replace:
        majority = {
            "job": maj_job,
            "education": maj_education,
            "contact": maj_contact,
            "poutcome": maj_poutcome
        }
    else:
        majority = None

    return S, medians, majority


# In[631]:


def create_attribute_dictionary(example_type, replace=False):
    """
    Creates a master dictionary for the attributes that are used for learning a decision tree.
    This is a convenience function and will vary for each type of problem.
    Currently this function supports two types of datasets:  "bank", and "tennis".
    The key is a string and represents a particular attribute in the sample data.
    The value is a tuple of strings and represents all possible values that the attribute can take.
    
    Inputs:
    -example_type:  "bank" -- represents the bank data example
                    "tennis" -- represents tennis game example used for testing implementation
    -replace:  If False, "unknown" attribute values are considered a value.  Otherwise if True, "unknown" 
               attribute values are replaced with the majority value of that attribute.
    Returns:
    -attributes:  a dictionary of attributes and values
    """
        
    if example_type == "bank":
        attributes = {
            "age": ("high", "low"),
            "marital": ("married", "divorced", "single"),
            "default": ("yes", "no"),
            "balance": ("high", "low"),
            "housing": ("yes", "no"),
            "loan": ("yes", "no"),
            "day":("high", "low"),
            "month":("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", 
                     "sep", "oct", "nov", "dec"),
            "duration": ("high", "low"),
            "campaign": ("high", "low"),
            "pdays": ("high", "low"),
            "previous": ("high", "low"),
            "label": ("yes", "no")
        }
        
        if not replace:
            attributes["job"] = ("admin.", "unknown", "unemployed", "management", "housemaid",
                    "entrepreneur", "student", "blue-collar", "self-employed", 
                    "retired", "technician","services")
            attributes["education"] = ("unknown", "secondary","primary","tertiary")
            attributes["contact"] = ("unknown","telephone","cellular")
            attributes["poutcome"] = ("unknown", "other", "failure", "success")
        else:
            attributes["job"] = ("admin.", "unemployed", "management", "housemaid",
                    "entrepreneur", "student", "blue-collar", "self-employed", 
                    "retired", "technician","services")
            attributes["education"] = ("secondary","primary","tertiary")
            attributes["contact"] = ("telephone","cellular")
            attributes["poutcome"] = ("other", "failure", "success")
        
    if example_type == "tennis":
        attributes = {
            "outlook":("sunny","overcast","rainy"),
            "temp":("hot","medium","cool"),
            "humidity":("high","normal","low"),
            "wind":("strong","weak"),
            "label":("yes","no")
        }
    
    return attributes


# In[581]:


def initialize_weights(S):
    """
    Creates weights associated with each example in S.  Initially, the weights are a uniform distribution.
    
    Input:
    -S:  A list of dictionaries with key-value pairs represented as strings.
    
    Output:
    Updates each entry in S to include an element in the dictionary with the key "weight".  The value of each
    key is the uniform distribution of the weights (1/m)
    """
        
    weight = 1 / float(len(S))
    for s in S:
        s["weight"] = weight

    


# In[477]:


def entropy(S):
    """
    Calculates the entropy of a given dataset S.
    
    Input:
    -S: A list of dictionaries with key-value pairs represented as strings. 
    
    Returns:
    -entropy:  A float, which is the calculated entropy for the key 'label'.
    """
    
    import math
    
    #Iterate through list to count the different labels
    counts = {}
    if len(S) == 0:
        return 0.0
    for s in S:
        value = s['label']
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1
    
    entropy = 0.0
    total_count = len(S)
    for count in counts.values():
        ratio = float(count) / total_count
        entropy += -1 * ratio * math.log(ratio,2)
    
    return entropy 
        


# In[543]:


def weighted_entropy(S):
    """
    Calculates the entropy of a given dataset S with each example in S associated with a weight
    
    Input:
    -S: A list of dictionaries with key-value pairs represented as strings. 
    
    Returns:
    -entropy:  A float, which is the calculated weighted entropy for the key 'label'.
    """
    
    import math
    
    #Iterate through list to count the different labels
    counts = {}
    if len(S) == 0:
        return 0.0
    
    total_weight = 0.0
    for s in S:
        value = s['label']
        if value in counts:
            counts[value] += s['weight']
        else:
            counts[value] = s['weight']
        
        total_weight += s['weight']
    
    entropy = 0.0
        
    for value in counts:
        ratio = float(counts[value]) / total_weight
        entropy += -1 * ratio * math.log(ratio,2)
    
    return entropy
        


# In[480]:


def majority_error(S):
    """
    Calculates the majority error of a given dataset S.
    
    Input:
    -S: A list of dictionaries with key-value pairs represented as strings. 
    
    Returns:
    -majority_error:  A float, which is the calculated majority error for the key 'label'.
    """
    
    import math
    
    #Iterate through list to count the different labels
    counts = {}
    if len(S) == 0:
        return 0.0
    for s in S:
        value = s['label']
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1
    majority = max(counts.values())
    total_count = len(S)
    majority_error = 1 - float(majority) / total_count
    return majority_error
        


# In[481]:


def gini_index(S):
    """
    Calculates the gini index for a given dataset S.
    
    Input:
    -S: A list of dictionaries with key-value pairs represented as strings. 
    
    Returns:
    -gini:  A float, which is the calculated gini index for the key 'label'.
    """
    
    import math
    
    #Iterate through list to count the different labels
    counts = {}
    if len(S) == 0:
        return 0.0
    for s in S:
        value = s['label']
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1
    
    total_count = len(S)
    gini = 0.0
    for count in counts.values():
        ratio = float(count) / total_count
        gini += ratio**2
    
    return 1 - gini
        


# In[573]:


def best_attribute(S, Attributes, master_list, error_type, weighted=False):
    """
    Determines the attribute A that produces the greatest information gain amongst a set of attributes.
    The information gain is determined by the error_type.
    
    Input:
    -S:  A list of examples, which are dictionaries of key-values represented attributes and values respectively.
    -Attributes:  A set of attributes that will be compared
    -master_list: A dictionary, which contains all the possible values each attribute can have
    -error_type:  One of three types:  "entropy", "me" (majority error) or "gini" (gini index)
    -weighted:  Boolean value that represents whether the data is weighted
    
    returns:
    -A:  a string that is the attribute with the largest information gain in the given dataset S. 
    """
    information_gain = {}
    
    #calculate the total weights for the normalizing constant
    if weighted:
        total_weight = 0.0
        for s in S:
            total_weight += s['weight']

    if error_type == "entropy":
        if weighted:
            current_entropy = weighted_entropy(S)
        else:
            current_entropy = entropy(S)
    if error_type == "me":
        current_entropy = majority_error(S)
    if error_type == "gini":
        current_entropy = gini_index(S)

    #Iterate through all attributes
    #If the sample s has the given value for the attribute, add it to the list
    for attribute in Attributes:
        expected_entropy = 0.0

        for value in master_list[attribute]:
            current_value = []
            for s in S:
                if s[attribute] == value:
                    current_value.append(s)
        
            if error_type == "entropy":
                value_entropy = entropy(current_value)
            if error_type == "me":
                value_entropy = majority_error(current_value)
            if error_type == "gini":
                value_entropy = gini_index(current_value)

            ratio = float(len(current_value)) / len(S)
                        
            if weighted:
                value_entropy = weighted_entropy(current_value)
                #calculate the total weight of the value 
                weight = 0.0
                for element in current_value:
                    weight += element['weight']
                ratio = weight / total_weight
   
            expected_entropy += ratio * value_entropy
        
        information_gain[attribute] = current_entropy - expected_entropy
        
    return max(information_gain, key=information_gain.get)
    


# In[640]:



def ID3(S, Attributes, master_list, error_type, current_depth, max_depth=float('inf'), weighted=False):
    """
    Creates a decision tree using the ID3 algorithm.
    
    Inputs: 
    -S: list of dictionaries; each dictionary contains a set of key-value pairs that are strings.
        The key is a string representing the attribute, and the value is a string representing the value of that
        attribute.  Labels are included as an attribute in the dictionary.  
        Each dictionary represents one example.
    -Attributes: set of attributes.  These are the attributes that will be searched when building the tree.
    -master_list: A dictionary, which contains all the possible values each attribute can have
    -error_type:  One of three types:  "entropy", "me" (majority error) or "gini" (gini index)
    -current_depth:  The current depth of the decision tree being constructed.
    -max_depth:  The maximum depth of the tree to be constructed.
    -weighted:  Boolean value that represents whether the data is weighted

    returns:
    -root_node:  A tree node
    """
    if current_depth == max_depth:
        label = majority_label(S,weighted=weighted)
        return Node(name='leaf', action=label)
    sample_size = len(S)
    
    #Test all labels to see if they are the same
    label = S[0]["label"]
    count = 0
    for s in S:
        if s["label"] != label:
            break
        else:
            count = count + 1
    
    if count == sample_size:
        
        #If attributes is empty, return a leaf node with the most common label
        if len(Attributes) == 0:
            label = majority_label(S, weighted=weighted)
        return Node(name='leaf', action=label)
    
    else:
        root_node = Node()
        A = best_attribute(S, Attributes, master_list, error_type, weighted)
        root_node.attribute = A
        if A in Attributes:
            Attributes.remove(A)
        
        for value in master_list[A]:
            
            #Create new subset of examples
            S_v = []
            for sample in S:
                if sample[A] == value:
                    S_v.append(sample)
            
            if len(S_v) == 0:
                maj_label = majority_label(S)
                new_node = Node(name="leaf", attribute=A, parent=root_node.attribute, action=maj_label)
                
            else:                
                new_node = ID3(S_v, Attributes, master_list, error_type, current_depth+1, max_depth, weighted)
                new_node.parent = root_node.attribute
            root_node.add_branch(value, new_node)
            
        #Add attribute removed from list so that next iteration of recursive call has the correct attribute set
        Attributes.add(A)
            
    return root_node
    


# In[555]:


def build_decision_tree(path, example, purity_type, max_depth=None, replace=False, weighted=False):
    """
    Creates a decision tree from the given dataset and parameters
    
    Inputs:
    -path:  A string, representing the path of the dataset to be processed.
    -example:  One of three types:  "bank", "tennis".  Represents the type of dataset to be used.
    -purity_type:  One of three types:  "entropy", "me" and "gini".  Represents how information gain will be
        calculated in the decision tree.
    -max_depth:  The maximum depth of the decision tree.  If None, then the tree will be the length of
        the total number of attributes
    -replace:  If False, "unknown" attribute values are considered a value.  Otherwise if True, "unknown" 
               attribute values are replaced with the majority value of that attribute.
    -weighted:  Boolean that represents whether the examples in the dataset are weighted.

               
    Returns:
    -tree:  A decision tree
    -medians:  A dictionary with key-value pairs.  The key is the attribute represented as a string and 
        the value is the corresponding numerical median.  This is only returned when "bank" is the example type.
        Otherwise, None is returned.
    -majority: A dictionary with key-value pairs .  The key is the attribute represented as a string and 
        the value is the corresponding majority element for that attribute.  This is only returned when "bank" 
        is the example type.  Otherwise, None is returned.

    """
    
    S = read_file(path, example)
    
    if max_depth is None:
        max_depth == len(S[0]) - 1
        print(max_depth)
    
    medians = majority = None
    if example == "bank":
        S, medians, majority = process_bank_data(S, "train", replace=replace)
    master_list = create_attribute_dictionary(example, replace)
    Attributes = set(list(master_list.keys()))
    Attributes.remove("label")
    
    return ID3(S, Attributes, master_list, purity_type, 0, max_depth, weighted=weighted), medians, majority


# In[604]:


def walk_tree(node, s):
    """
    Recurisive function.  Given an example s walks through a tree, following the branch of each 
    node with the given value in s.  
    
    Inputs:  
    -node:  A tree node.
    -s: One data instance containing attributes and vaues.
    
    Return:
    -action:  A string, representing the action of the leaf node.
    """
    
    value = s[node.attribute]
    next_node = node.get_branch(value)
    if next_node.name == "root":
        action = walk_tree(next_node, s)
    else:
        #At leaf node
        action = next_node.action
    
    return action


def test_decision_tree(tree, S):
    """
    With a given decision tree, tests the decision tree for accuracy against a given data set.
    
    Inputs:
    -tree:  The decision tree to be tested.
    -S:  A testing set
               
    Returns:
    -ratio:  A float, representing the error rate of the decision tree with the given dataset.
    """
    
    S = read_file(path, example)
    
    #itereate through each item of list.  Keep track of correct classifications and incorrect classfications
    correct = incorrect = 0
    count = 1
    for s in S:
        action = walk_tree(tree, s)
        if action == s["label"]:
            correct += 1
        else:
            incorrect += 1
        count += 1
            
    #print("The number correct is " + str(correct) + " and the number incorrect is " + str(incorrect))
    ratio = float(correct) / len(S)
    return ratio


# In[641]:


def update_weights(S, alpha, h):
    """
    Updates the weights in dataset S
    
    Input:
    -S:  A training set, which is list of dictionaries with key-value pairs representing attributes and values
    -alpha:  Float, which represents the weight for the given hypothesis
    -h:  A decision tree, which represents the given hypothesis
    """
    
    import math
    
    #Calculate Normalizing Constant
    Z = 0.0
    
    for s in S:
        action = walk_tree(h, s)
        multiplier = 1.0
        if action != s["label"]:
            multiplier = -1.0
        updated_weight = s['weight']  * math.exp(-1 * alpha * multiplier)
        s['weight'] = updated_weight

    for s in S:
        Z += s["weight"]

        
    for s in S:
        s['weight'] = s['weight'] / Z
        
    


# In[642]:


def calculate_weighted_error(S, h):
    """
    Calculates the error in a dataset S where each sample has a corresponding weight
    
    Input:
    -S:  A dataset containing examples.  
    -h:  A decision tree, which represents a hypothesis to be tested
    
    Returns:
    weighted_sum:  A float which is the error rate of the given hypothesis.
    """
    
    weighted_sum = 0.0
    for s in S:
        action = walk_tree(h, s)
        if action != s["label"]:
            weighted_sum += s['weight']
            
    return weighted_sum
        


# In[619]:


def calculate_unweighted_error(S, h):
    """
    Calculates the error in a dataset S where each sample has a corresponding weight
    
    Input:
    -S:  A dataset containing examples.  
    -h:  A decision tree, which represents a hypothesis to be tested
    
    Returns:
    weighted_sum:  A float which is the error rate of the given hypothesis.
    """
    
    incorrect = 0;
    for s in S:
        action = walk_tree(h, s)
        if action != s["label"]:
            incorrect +=1
    ratio = float(incorrect)/len(S)
    #print("The total inccorect are " + str(incorrect) + " out of " + str(len(S)) + 
       #   ' for a total ratio of ' + str(ratio))
            
    return ratio
    
        


# In[620]:


def adaBoost(S, master_list, T):
    """
    Inputs:
    -S: list of dictionaries; each dictionary contains a set of key-value pairs that are strings.
        The key is a string representing the attribute, and the value is a string representing the value of that
        attribute.  Labels are included as an attribute in the dictionary.  
        Each dictionary represents one example.
    -master_list: A dictionary, which contains all the possible values each attribute can have

    -T: number of iterations or hypothesis to generate in the ensemble
    
    Returns:
    -ensemble:  A list of tuples.  Each tuple contains is a (hypotheses, weight).
    """
    
    import math
    
    attributes = set(list(master_list.keys()))
    attributes.remove("label")
    
    initialize_weights(S)
    ensemble = []
    Z = 1.0
    alpha = 0.0
    for i in range(T):
        stump = ID3(S, attributes, master_list, "entropy", 0, 2, weighted=True)
        error = calculate_weighted_error(S, stump)
        alpha = 0.5 * math.log((1 - error)/error)
        ensemble.append((stump, alpha))
        update_weights(S, alpha, stump)
            

        #print("This is the the " + str(i) +"th iteration of adaBoost and the error rate is " + str(error)
         #    + " and alpha is " + str(alpha))
    
    return ensemble
        
        


# In[643]:


def ensemble_result(s, ensemble):
    """
    Calculates the binary result of a weighted ensemble
    
    Input:
    -s:  A single sample
    -ensemble:  A list of tuples containing (hypothesis, weights)
    
    Returns:
    -label: "yes" if the total weighted sum > 0; otherwise "no" 
    """

        
    h_sum = 0.0
    for item in ensemble:
        action = walk_tree(item[0], s)
        if action == "yes":
            multiplier = 1
        else:
            multiplier = -1
        h_sum += item[1] * multiplier
        
    if h_sum > 0:
        return "yes"
    else:
        return "no"
    


# In[622]:


def test_ensemble(ensemble, S):
    """
    Tests the accuracy of the given ensemble.
    
    Inputs:
    -ensemble:  A list of tuples.  Each tuple is (hypothesis, weight) in the given ensemble.
    -S:  The testing dataset
    
    Returns:
    -ratio:  A float, representing the error rate of the ensemble with the given dataset.
    """
    
    #itereate through each item of list.  Keep track of correct classifications and incorrect classfications
    correct = incorrect = 0
    for s in S:
        action = ensemble_result(s, ensemble)
        if action == s["label"]:
            correct += 1
        else:
            incorrect += 1
            
    #print("The number correct is " + str(correct) + " and the number incorrect is " + str(incorrect))
    ratio = float(correct) / len(S)
    return ratio


# In[623]:


def calculate_error(S, h):
    """
    Calculates the error in a dataset S.
    
    Input:
    -S:  A dataset containing examples.  
    -h:  A decision tree, which represents a hypothesis to be tested
    
    Returns:
    error:  A float which is the error rate of the given hypothesis.
    """
    
    tot_incorrect = 0
    for s in S:
        action = walk_tree(h, s)
        if action != s["label"]:
            tot_incorrect += 1
            
    return float(tot_incorrect) / len(S)
    
        


# In[624]:


def test_decision_stumps(S, ensemble):
    """
    Input:
    -S:  A dataset containing examples.  
    -ensemble:  A list of tuples.  Each tuple is (hypothesis, weight) in the given ensemble.
    
    Returns:
    -errors:  list of error rates for each decision stump.  It is the length of the ensemble.
    """
    
    errors = []
    #print(ensemble)
    for item in ensemble:
        errors.append(calculate_error(S, item[0])*100)
    
    return errors


# In[645]:


def test_adaboost(T=1000):
    """
    Method for testing Adaboost algorithm with bank dataset.
    Problem #2.2a in HW2 for CS3505
    T - Number of times algorithm will be run
    """
    
    import matplotlib.pyplot as plt

    S_train = read_file('train.csv', "bank")
    S_train, medians, majority = process_bank_data(S_train, "train")
    S_test = read_file('test.csv', "bank")
    S_test, medians, _ = process_bank_data(S_test, "test", medians)
    master_list = create_attribute_dictionary("bank")

    training_errors = []
    testing_errors = []

    for i in range(T):
        ensemble = adaBoost(S_train, master_list, i)
        training_errors.append(100 - test_ensemble(ensemble, S_train)*100)
        testing_errors.append(100 - test_ensemble(ensemble, S_test)*100)

    #plot the error rates versus the no of iterations for train and test sets
    plt.title('Error rates per no. of iterations T\nAdaBoost')
    plt.xlabel('Iteration')
    plt.ylabel('Percentage Incorrect')
    plt.plot(training_errors, label="train")
    plt.plot(testing_errors, label="test")
    plt.legend(loc='lower right')
    plt.show()

    #plot the error rates of each decision stump for train and test sets
    stump_errors_train = test_decision_stumps(S_train, ensemble)
    stump_errors_test = test_decision_stumps(S_test, ensemble)
    plt.title('Error rates for decision stumps\nAdaBoost')
    plt.xlabel('Decision stump no.')
    plt.ylabel('Percentage Incorrect')
    plt.plot(stump_errors_train, label="train")
    plt.plot(stump_errors_test, label="test")
    plt.legend(loc='lower right')
    plt.show()


# In[463]:


def bagged_trees(S, master_list, T, m=None):
    """
    Inputs:
    -S: list of dictionaries; each dictionary contains a set of key-value pairs that are strings.
        The key is a string representing the attribute, and the value is a string representing the value of that
        attribute.  Labels are included as an attribute in the dictionary.  
        Each dictionary represents one example.
    -master_list: A dictionary, which contains all the possible values each attribute can have

    -T: number of iterations or hypothesis to generate in the ensemble
    -m:  size of the sample set.  Defaults to the size of the dataset S.
    
    Returns:
    -ensemble:  A list of tuples.  Each tuple is a (tree, weight). Each are equally weighted.
    
    """
    import random

    attributes = set(list(master_list.keys()))
    attributes.remove("label")
    
    ensemble = []
    for i in range(T):
        
        #Draw new samples
        new_samples = []
        if m is None:
            m = len(S)
        for j in range(m):
            new_samples.append(S[random.randint(0,len(S)-1)])

        tree = ID3(new_samples, attributes, master_list, "entropy", 0, 16)
        ensemble.append((tree, 1.0))
    
    return ensemble
    


# In[464]:


def test_bagging(T=1000):
    """
    Method for testing the bagging algorithm with bank dataset.
    Problem #2.2b in HW2 for CS3505
    T - number of times algorith will be run
    """
    import matplotlib.pyplot as plt

    S_train = read_file('train.csv', "bank")
    S_train, medians, majority = process_bank_data(S_train, "train")
    S_test = read_file('test.csv', "bank")
    S_test, medians, _ = process_bank_data(S_test, "test", medians)
    master_list = create_attribute_dictionary("bank")

    training_errors = []
    testing_errors = []
    for i in range(T):
        ensemble = bagged_trees(S_train, master_list, i, 500)
        training_errors.append(100 - test_ensemble(ensemble, S_train)*100)
        testing_errors.append(100 - test_ensemble(ensemble, S_test)*100)

    #plot the error rates versus the no of trees
    plt.title('Error rates per no of trees\nBagged Trees')
    plt.xlabel('No. of Trees')
    plt.ylabel('Percentage Incorrect')
    plt.plot(training_errors, label="train")
    plt.plot(testing_errors, label="test")
    plt.legend(loc='lower right')
    plt.yticks(np.arange(0, 20, 5))
    plt.show()



# In[230]:


def calculate_avg_pred_single_tree(s, tree_list):
    """
    Inputs:
    -s:  a single example
    -tree_list:  a list of decision trees. 
    
    Returns:
    -avg_prediction:  The average prediction of the all the tree in the tree list for example s.  
        1 represents "yes" and -1 represents "no".  The average prediction will be a float in between -1 and 1.
    """
    
    prediction = 0
    for tree in tree_list:
        action = walk_tree(tree, s)
        if action == s["label"]:
            prediction += 1
        else:
            prediction += -1
            
    avg_prediction = float(prediction) / len(tree_list)
    return avg_prediction
        


# In[231]:


def calculate_avg_pred_ensemble(s, ensemble_list):
    """
    Inputs:
    -s:  a single example
    -ensemble_list:  a list of ensembles.  Each ensemble represents a list of tuples of (hypothesis, weight).
    
    Returns:
    -avg_prediction:  The average prediction of the all the ensembles in the ensemble list for example s.  
        1 represents "yes" and -1 represents "no".  The average prediction will be a float in between -1 and 1.
    """
    
    total_avg_pred = 0.0
    for ensemble in ensemble_list:
        prediction = 0;
        for tree in ensemble:
            action = walk_tree(tree[0], s)
            if action == s["label"]:
                prediction += 1
            else:
                prediction += -1
        avg_prediction = float(prediction) / len(ensemble)
        total_avg_pred += avg_prediction
    
    total_avg_pred = total_avg_pred / len(ensemble_list)
    return total_avg_pred
        


# In[232]:


def calculate_variance_single_tree(s, tree_list, avg_pred):
    
    """
    Inputs:
    -s:  a single example
    -tree_list:  a list of decision trees
    -avg_prediction:  The average prediction of the trees in the tree list.  1 represents "yes" and -1 represents "no".
        The average prediction will be a float in between -1 and 1.
    
    Returns:
    -variance:  A float representing the average variance of all the predictions
    """
    
    variance = 0.0
    for tree in tree_list:
        action = walk_tree(tree, s)
        prediction = 1
        if action != s["label"]:
            prediction = -1
        var = (avg_pred - prediction)**2
        variance += var
    
    variance = var / float(1)/(len(tree_list)-1)
    return variance


# In[233]:


def calculate_variance_ensemble(s, ensemble_list, avg_pred):
    """
    Inputs:
    -s:  a single example
    -ensemble_list:  a list of ensembles.  Each ensemble represents a list of tuples of (hypothesis, weight).
    -avg_prediction:  The average prediction of the all the ensembles in the ensemble list for example s.  
        1 represents "yes" and -1 represents "no".  The average prediction will be a float in between -1 and 1.
    
    Returns:
    -variance:  A float representing the average variance of all the predictions in the ensemble_list
    """ 
    
    variance = 0.0
    avg_variance = 0.0
    for ensemble in ensemble_list:
        for tree in ensemble:
            action = walk_tree(tree[0], s)
            prediction = 1
            if action != s["label"]:
                prediction = -1
            var = (avg_pred - prediction)**2
            variance += var
    
        variance = var / float(1)/(len(ensemble)-1)
        avg_variance += variance
    
    avg_variance = avg_variance / len(ensemble_list)
    return variance


# In[277]:


def calculate_bv_bagged(T=1000):
    #Calculate bias and variance for test samples for bagged ensembles and single decision trees  
    #Problem 2.2c in HW2
    #T - number of times algorithm will be run
    import random

    S_train = read_file('train.csv', "bank")
    S_train, medians, majority = process_bank_data(S_train, "train")
    S_test = read_file('test.csv', "bank")
    S_test, medians, _ = process_bank_data(S_test, "test", medians)
    master_list = create_attribute_dictionary("bank")

    #list of lists of ensemble tuples(hypothesis, weight=1.0)
    bagged_ensembles = [] 

    #list of hypothesis
    single_trees = []
    for i in range(T):
        
        ensemble = bagged_trees(S_train, master_list, T, 500)                     
        bagged_ensembles.append(ensemble) 
        single_trees.append(ensemble[0][0])

    avg_bias = 0.0
    avg_var = 0.0
    
    for s in S_test: 
        avg_pred = calculate_avg_pred_single_tree(s, single_trees)
        correct_pred = 1
        if s['label'] == 'no':
            correct_pred = -1
        bias = (avg_pred - correct_pred)**2
        variance = calculate_variance_single_tree(s, single_trees, avg_pred)
    
        avg_bias += bias
        avg_var += variance

    avg_bias = avg_bias / len(S_test)
    avg_var = avg_var / len(S_test)

    print("The average bias for the single trees is " + '{:.6f}'.format(avg_bias) + " and the average variance is "
        + '{:.6f}'.format(avg_var) + " for a total estimated general squared error of " + '{:.6f}'.format(avg_bias + avg_var))
    print("\n")

    avg_bias = 0.0
    avg_var = 0.0
    for s in S_test: 
        avg_pred = calculate_avg_pred_ensemble(s, bagged_ensembles)
        correct_pred = 1
        if s['label'] == 'no':
            correct_pred = -1
        bias = (avg_pred - correct_pred)**2
        variance = calculate_variance_ensemble(s, bagged_ensembles, avg_pred)
    
        avg_bias += bias
        avg_var += variance

    avg_bias = avg_bias / len(S_test)
    avg_var = avg_var / len(S_test)

    print("The average bias for the bagged trees is " + '{:.6f}'.format(avg_bias) + " and the average variance is "
        + '{:.6f}'.format(avg_var) + " for a total estimated general squared error of " + '{:.6f}'.format(avg_bias + avg_var))
    print("\n")

    
    

        


# In[279]:


def RandTreeLearn(S, Attributes, master_list, error_type, current_depth, sample_size=2, max_depth=float('inf')):    
    """
    Creates a decision tree using the ID3 algorithm.
    
    Inputs: 
    -S: list of dictionaries; each dictionary contains a set of key-value pairs that are strings.
        The key is a string representing the attribute, and the value is a string representing the value of that
        attribute.  Labels are included as an attribute in the dictionary.  
        Each dictionary represents one example.
    -Attributes: set of attributes.  These are the attributes that will be searched when building the tree.
    -master_list: A dictionary, which contains all the possible values each attribute can have
    -error_type:  One of three types:  "entropy", "me" (majority error) or "gini" (gini index)
    -current_depth:  The current depth of the decision tree being constructed.
    -sample_size:  The size of the attribute sample that will be selected randomly
    -max_depth:  The maximum depth of the tree to be constructed.

    returns:
    -root_node:  A tree node
    """
    
    import random

    if current_depth == max_depth:
        label = majority_label(S)
        return Node(name='leaf', action=label)
    sample_size = len(S)
    
    #Test all labels to see if they are the same
    label = S[0]["label"]
    count = 0
    for s in S:
        if s["label"] != label:
            break
        else:
            count = count + 1
    
    if count == sample_size:
        
        #If attributes is empty, return a leaf node with the most common label
        if len(Attributes) == 0:
            label = majority_label(S)
        return Node(name='leaf', action=label)
    
    else:
        root_node = Node()
        
        #choose random attributes to split on
        if len(Attributes) < sample_size:
            k = len(Attributes)
        else:
            k = sample_size
        rand_attr = set(random.sample(Attributes, k))
        
        A = best_attribute(S, rand_attr, master_list, error_type)
        root_node.attribute = A
        if A in Attributes:
            Attributes.remove(A)
        
        for value in master_list[A]:
            
            #Create new subset of examples
            S_v = []
            for sample in S:
                if sample[A] == value:
                    S_v.append(sample)
            
            if len(S_v) == 0:
                maj_label = majority_label(S)
                new_node = Node(name="leaf", attribute=A, parent=root_node.attribute, action=maj_label)
                
            else:                
                new_node = ID3(S_v, Attributes, master_list, error_type, current_depth+1, max_depth)
                new_node.parent = root_node.attribute
            root_node.add_branch(value, new_node)
            
        #Add attribute removed from list so that next iteration of recursive call has the correct attribute set
        Attributes.add(A)
            
    return root_node
    


# In[280]:


def random_forest(S, master_list, T, sample_size):
    """
    Inputs:
    -S: list of dictionaries; each dictionary contains a set of key-value pairs that are strings.
        The key is a string representing the attribute, and the value is a string representing the value of that
        attribute.  Labels are included as an attribute in the dictionary.  
        Each dictionary represents one example.
    -master_list: A dictionary, which contains all the possible values each attribute can have
    -T: number of iterations or hypothesis to generate in the ensemble
    -sample_size:  The size of the attribute sample that will be selected randomly
    
    Returns:
    -ensemble:  A list of tuples.  Each tuple is a (tree, weight). Each are equally weighted.
    
    """
    import random
    
    attributes = set(list(master_list.keys()))
    attributes.remove("label")
    
    ensemble = []
    for i in range(T):
        
		#Draw new samples
        new_samples = []
        for j in range(len(S)):
            new_samples.append(S[random.randint(0,len(S)-1)])
	
        tree = RandTreeLearn(new_samples, attributes, master_list, "entropy", 0, sample_size=sample_size, max_depth=16)
        ensemble.append((tree, 1.0))
    
    return ensemble
    


# In[281]:


def test_random_forest(T=1000):
    """
    Method for testing the random forest algorithm with bank dataset.
    Problem #2.2d in HW2 for CS3505
    T = number of times algorithm will be run
    """
    import matplotlib.pyplot as plt

    S_train = read_file('train.csv', "bank")
    S_train, medians, majority = process_bank_data(S_train, "train")
    S_test = read_file('test.csv', "bank")
    S_test, medians, _ = process_bank_data(S_test, "test", medians)
    master_list = create_attribute_dictionary("bank")


    for s in (2,4,6):
        training_errors = []
        testing_errors = []
        for i in range(T):
            ensemble = random_forest(S_train, master_list, i, s)
            training_errors.append(100 - test_ensemble(ensemble, S_train)*100)
            testing_errors.append(100 - test_ensemble(ensemble, S_test)*100)

        #plot the error rates versus the no of trees
        print("Sample size: " + str(s))
        plt.title('Error rates per no of trees\nRandom Forest')
        plt.xlabel('No. of Trees')
        plt.ylabel('Percentage Incorrect')
        plt.plot(training_errors, label="train")
        plt.plot(testing_errors, label="test")
        plt.legend(loc='lower right')
        plt.yticks(np.arange(0, 20, 5))
        plt.show()


# In[282]:


def calculate_bv_random_forest(T=1000):
    #Calculate bias and variance for test samples for random forest ensembles and single decision trees
    #Problem 2.2e in HW2
    #T=number of times algorithm will be run
    import random

    S_train = read_file('train.csv', "bank")
    S_train, medians, majority = process_bank_data(S_train, "train")
    S_test = read_file('test.csv', "bank")
    S_test, medians, _ = process_bank_data(S_test, "test", medians)
    master_list = create_attribute_dictionary("bank")

    #list of lists of ensemble tuples(hypothesis, weight=1.0)
    random_ensembles = [] 
    single_trees = []

    #list of hypothesis
    for i in range(T):
        
        ensemble = random_forest(S_train, master_list, T, 4)                     
        random_ensembles.append(ensemble) 
        single_trees.append(ensemble[0][0])

    avg_bias = 0.0
    avg_var = 0.0

    for s in S_test: 
        avg_pred = calculate_avg_pred_single_tree(s, single_trees)
        correct_pred = 1
        if s['label'] == 'no':
            correct_pred = -1
        bias = (avg_pred - correct_pred)**2
        variance = calculate_variance_single_tree(s, single_trees, avg_pred)
        
        avg_bias += bias
        avg_var += variance

    avg_bias = avg_bias / len(S_test)
    avg_var = avg_var / len(S_test)

    print("The average bias for the single trees is " + '{:.6f}'.format(avg_bias) + " and the average variance is "
        + '{:.6f}'.format(avg_var) + " for a total estimated general squared error of " + '{:.6f}'.format(avg_bias + avg_var))
    print("\n")

    avg_bias = 0.0
    avg_var = 0.0
    for s in S_test: 
        avg_pred = calculate_avg_pred_ensemble(s, random_ensembles)
        correct_pred = 1
        if s['label'] == 'no':
            correct_pred = -1
        bias = (avg_pred - correct_pred)**2
        variance = calculate_variance_ensemble(s, random_ensembles, avg_pred)
    
        avg_bias += bias
        avg_var += variance

    avg_bias = avg_bias / len(S_test)
    avg_var = avg_var / len(S_test)

    print("The average bias for the random forest trees is " + '{:.6f}'.format(avg_bias) + " and the average variance is "
        + '{:.6f}'.format(avg_var) + " for a total estimated general squared error of " + '{:.6f}'.format(avg_bias + avg_var))
    print("\n")
    

        


# In[283]:


#Run all the Homework Problems
test_adaboost(10)
test_bagging(10)
test_random_forest(10)
calculate_bv_bagged(10)
calculate_bv_random_forest(10)



