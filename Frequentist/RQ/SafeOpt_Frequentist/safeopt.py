import torch
import gpytorch
import gp_model as gpm
import random

def classify_safe_set_old(lower, flim):
        
        safe_set = (lower > flim)
        unsafe_set = (lower < flim)

        return safe_set, unsafe_set


def classify_safe_set(model, test_x, jmin):
    safe_set = torch.zeros(len(test_x), dtype=torch.bool)

    for i, x in enumerate(test_x):
        lower = gpm.get_bounds_RKHS(model, x.unsqueeze(0))[1]

        if lower.item() > jmin:
            safe_set[i] = True

    return safe_set



def get_maximiser_ic(test_x, safe_set, max_set):
    # Identify positions where both safe_set and max_set are True
    valid_indices = torch.nonzero(safe_set & max_set, as_tuple=False).squeeze()
    
    # If no such indices exist, return None
    if valid_indices.numel() == 0:
        return None
    
    # Choose a random index from the valid indices
    random_index = random.choice(valid_indices.tolist())
    
    # Return the value from test_x at the chosen index
    x_max = test_x[random_index].item()
    x_max = torch.tensor([[[x_max]]])

    return x_max




# def classify_expander(model, test_x, jmin):
#     mean, lower, upper = gpm.get_bounds_RKHS(model, test_x)
#     # safe_set = classify_safe_set(lower, jmin)[0]
#     safe_set = classify_safe_set(model, test_x, jmin)
#     expander_set = torch.empty(0)

#     for i in range(upper.size(0)):
#         x_add = test_x[i]
#         y_add = upper[i]
        
#         train_x = model.train_inputs[0]
#         targets_y = model.train_targets
#         train_x_add = torch.vstack([train_x, x_add])
#         train_y_add = torch.cat([targets_y, y_add])
#         model_add = gpm.init_model(train_x_add, train_y_add)
#         # model_add = gpm.create_singletask_gp(train_x_add, train_y_add)
     
#         model_add.eval()

#         lower_add = gpm.get_bounds_RKHS(model_add, test_x)
#         # safe_set_add = classify_safe_set(lower_add, jmin)[0]
#         safe_set = classify_safe_set(model, test_x, jmin)

#         if safe_set_add.size(0) > safe_set.size(0):
#             expander_set = torch.cat((expander_set, x_add))

#         return expander_set

def get_expander_ic_mod():

    a = float(input("Enter value for x_safe: "))
    b = float(input('Enter value for x_unsafe: '))
    x0_exp = torch.tensor([[a,b]])
    
    return x0_exp


def get_expander_ic(model, test_x, safe_set):
    # Find all pairs where one value is safe and the next is unsafe, or vice versa
    pairs = torch.empty((0, 4), dtype=torch.float32)

    for i in range(safe_set.size(0) -1 ):
        if safe_set[i] != safe_set[i+1]:
            if safe_set[i]:
                new_pair = torch.tensor([i, test_x[i].item(), i+1, test_x[i+1].item()])
            else:
                new_pair = torch.tensor([i+1, test_x[i+1].item(), i, test_x[i].item()])
            pairs = torch.cat((pairs, new_pair.unsqueeze(0)), dim=0)

    if pairs.size(0) == 0:
        print('ERROR: NO EXPANDER IC FOUND!')
        return None


    chosen_pair = pairs[random.randint(0, pairs.size(0) -1)] 
    x_safe = chosen_pair[1].item()
    x_unsafe = chosen_pair[3].item()

    x_exp = torch.tensor([[[x_safe, x_unsafe]]])   

    return x_exp


def find_max_maximiser(safe_set, un, ln):
    # find largest safe lower bound l*
    ln_max = torch.max(ln)
    # find all upper bounds larger than l* : pot_max
    pot_max = un > ln_max

    # if empty, return none 
    if len(pot_max) == 0:
        return None
    
    # compute uncertainty
    wn = un - ln

    # set uncertainties to zero, if corresponding upper bound smaller than l*
    for i in range(len(wn)):
        if pot_max[i] == False:
            wn[i] = 0
        if safe_set[i] == False:
            wn[i] = 0

    # find index of largest maximiser
    x_max1_idx = torch.argmax(wn)

    return x_max1_idx

def classify_maximizer(model, test_x):

    lower = gpm.get_bounds_RKHS(model, test_x)[1]
    lower_max = torch.max(lower)

    max_set = lower >= lower_max

    for i, x in enumerate(test_x):
        upper = gpm.get_bounds_RKHS(model, x.unsqueeze(0))[2]

        if upper.item() > lower_max:
            max_set[i] = True

    return max_set



def classify_enlarger(train_x, safe_set, upper_bounds, L, flim):
    enlarge_safe_set = []

    safe_indices = torch.nonzero(safe_set, as_tuple=True)[0]
    unsafe_indices = torch.nonzero(~safe_set, as_tuple=True)[0]

    for safe_idx in safe_indices:
        min_dist = float('inf')
        for unsafe_idx in unsafe_indices:
            dist = torch.dist(train_x[safe_idx], train_x[unsafe_idx])
            min_dist = min(min_dist, dist)

        g = upper_bounds[safe_idx] - L * min_dist

        if g >= flim:
            enlarge_safe_set.append(train_x[safe_idx].item())

    return enlarge_safe_set





def sample_criterion(model, likelihood, max_set, enlarge_safe_set):
    # Combine max_set and enlarge_safe_set
    combined_set = max_set + enlarge_safe_set
    
    # Initialize variables to keep track of the best point and maximum width
    best_x = None
    max_width = -float('inf')
   
    for x in combined_set:
        # Convert the point to a tensor 
        x_tensor = torch.tensor([x])
       
        width = compute_confidence_width(model, likelihood, x_tensor)
        
        # Check if this width is greater than the current maximum width
        if width > max_width:
            max_width = width
            best_x = x
    
    return best_x

