import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score, accuracy_score

def set_random_state(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_loss(features, target, coefficients, regularization_coefficient):
    logits = np.dot(features, coefficients)    
    logits_sigmoid = sigmoid(logits)        
    loss = (- np.sum(target*np.log(logits_sigmoid)+ (1-target)*np.log(1-logits_sigmoid)) \
            + 0.5*regularization_coefficient*np.dot(coefficients, coefficients))/len(target)    
    return loss

def train_one_iteration_dp_laplace(features, target, coefficients, learning_rate, 
                                    regularization_coefficient, epsilon=0.01):
    f_x = np.dot(features, coefficients)
    p_x = sigmoid(f_x)  
    gradient = (np.dot(features.T, (p_x - target)) + regularization_coefficient * coefficients)/features.shape[0]
    sensitivity = 2 / features.shape[0]
    beta = sensitivity / epsilon
    laplace_noise = np.random.laplace(0, beta, gradient.shape[0])
    gradient_new = gradient + laplace_noise
    coefficients -= learning_rate * gradient_new        
    return coefficients, gradient

def train_one_iteration_dp_gaussian(features, target, coefficients, learning_rate, 
                                    regularization_coefficient, epsilon=0.01, delta=1e-5):
    sigma = (2/features.shape[0]) * np.sqrt(2*np.log(1.25/delta)) / epsilon
    gaussian_noise = np.random.normal(loc=0.0, scale=sigma*sigma, size=coefficients.shape[0])        
    f_x = np.dot(features, coefficients)
    p_x = sigmoid(f_x)  
    gradient = (np.dot(features.T, (p_x - target)) + regularization_coefficient * coefficients)/features.shape[0]
    gradient_new = gradient + gaussian_noise
    coefficients -= learning_rate * gradient_new        
    return coefficients, gradient

def train_one_iteration_non_dp(features, target, coefficients, learning_rate, regularization_coefficient):
    f_x = np.dot(features, coefficients)
    p_x = sigmoid(f_x)  
    gradient = (np.dot(features.T, (p_x - target)) + regularization_coefficient * coefficients)/features.shape[0]
    coefficients -= learning_rate * gradient    
    return coefficients, gradient

def logistic_regression(features_train, target_train, features_validation, target_validation, add_x0,
                        iterations, learning_rate, regularization_coefficient, method='non_dp', epsilon=0.01, delta=1e-5):
    set_random_state(42)
    
    if add_x0:
        x0 = np.ones((features_train.shape[0], 1))
        features_train = np.hstack((x0, features_train))        
        x0 = np.ones((features_validation.shape[0], 1))
        features_validation = np.hstack((x0, features_validation))    
    coefficients = np.zeros(features_train.shape[1])
       
    best_coefficients = np.zeros(features_train.shape[1])
    best_loss = 9999
    count = 0
    patience = 100
    best_iteration = 0
    
    all_train_loss=[]
    all_validation_loss=[]
    all_gradients=[]
    progess_bar = tqdm(total=iterations, unit='iterations')
    for step in range(iterations):
        if method == 'non_dp':
            coefficients, gradient = train_one_iteration_non_dp(features_train, target_train, coefficients, learning_rate, regularization_coefficient)
        elif method == 'gaussian':
            coefficients, gradient = train_one_iteration_dp_gaussian(features_train, target_train, coefficients, learning_rate, regularization_coefficient, epsilon, delta)
        elif method == 'laplace':
            coefficients, gradient = train_one_iteration_dp_laplace(features_train, target_train, coefficients, learning_rate, regularization_coefficient, epsilon)
        
        train_loss = calculate_loss(features_train, target_train, coefficients, regularization_coefficient)
        validation_loss = calculate_loss(features_validation, target_validation, coefficients, regularization_coefficient)
        
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_coefficients = coefficients.copy()
            count = 0
            best_iteration = step
        else:
            count += 1
        
        progess_bar.set_postfix({'Train loss':train_loss, ' Validation_loss':validation_loss})
        progess_bar.update(1)
        
        all_train_loss.append(train_loss)
        all_validation_loss.append(validation_loss)
        all_gradients.append(gradient)
        
        if count == patience:
            learning_rate = learning_rate * 0.5
            count = 0
            
    progess_bar.close()  

    all_gradients = np.stack(all_gradients)
    all_gradients_norm = np.linalg.norm(all_gradients, axis=-1)  
    
    return best_coefficients, all_train_loss, all_validation_loss, all_gradients_norm, best_iteration

def get_data(do_norm=False, norm_order=2):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=512)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512)
    
    X_all = []; y_all = [];
    
    for _, (X,y) in enumerate(train_loader):
        X_all.append(X.squeeze(1).view(X.size(0), -1).data.numpy())
        y_all.append(y.data.numpy())
    
    for _, (X,y) in enumerate(test_loader):
        X_all.append(X.squeeze(1).view(X.size(0), -1).data.numpy())
        y_all.append(y.data.numpy())
    
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    
    X_all = X_all[np.logical_or(y_all == 5, y_all == 8)]
    y_all = y_all[np.logical_or(y_all == 5, y_all == 8)]
    y_all[y_all == 5] = 0
    y_all[y_all == 8] = 1
    
    if do_norm:
        X_all_norm = np.linalg.norm(X_all, norm_order, axis=-1)
        X_all_norm_max = np.max(X_all_norm)
        X_all = X_all / X_all_norm_max
    
    X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.10, random_state=42)
    
    return X_train, y_train, X_validation, y_validation, X_test, y_test

def train(add_x0=True, iterations=5000, learning_rate=0.5, regularization_coefficient=0.001, method='non_dp',
          epsilon=0.01, delta=1e-5, data_norm=False, norm_order=2):
    
    X_train, y_train, X_validation, y_validation, X_test, y_test = get_data(data_norm, norm_order)
    best_coefficients, all_train_loss, all_validation_loss, all_gradients_norm, _ = logistic_regression(
                                                                                            X_train, y_train, 
                                                                                            X_validation, y_validation, 
                                                                                            add_x0,
                                                                                            iterations, 
                                                                                            learning_rate,                                                                                                        
                                                                                            regularization_coefficient,
                                                                                            method,
                                                                                            epsilon,
                                                                                            delta,
                                                                                            )
    
    os.makedirs('./logs/', exist_ok=True)
    # plot of 5a. train and validation loss vs iterations
    iterations = np.arange(0, iterations)
    plt.plot(iterations, all_train_loss, label='train_loss')
    plt.plot(iterations, all_validation_loss, label='validation_loss')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('loss vs iterations [5a]')
    plt.savefig("./logs/{}_5a_loss_vs_iterations.png".format(method), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # plot of 5b. Norm of the gradient vs iterations
    plt.plot(iterations, all_gradients_norm)    
    plt.xlabel('iterations')
    plt.ylabel('Norm of the gradient')
    plt.title('Norm of the gradient vs iterations [5b]')
    plt.savefig("./logs/{}_5b_Norm_of_the_gradient_vs_iterations.png".format(method), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    if add_x0:
        X_test = np.hstack((np.ones((X_test.shape[0], 1)),X_test))
    logits = np.dot(X_test, best_coefficients)
    prediction_probabilites = sigmoid(logits)
    
    all_precisions=[]
    all_recalls=[]
    all_thresholds=np.arange(0, 1, 0.01)
    for threshold in all_thresholds:
        y_true = y_test.copy()
        y_pred = prediction_probabilites >= threshold
        all_precisions.append(precision_score(y_true, y_pred, average='macro'))
        all_recalls.append(recall_score(y_true, y_pred, average='macro'))
    
    # plot of 5f. Precision and Recall vs threshold
    plt.plot(all_thresholds, all_precisions, label='Precision')
    plt.plot(all_thresholds, all_recalls, label='Recall')
    plt.legend()
    plt.xlabel('Thresholds')
    plt.ylabel('Precision and Recall Score')
    plt.title('Precision and Recall Score vs Thresholds [5f]')
    plt.savefig("./logs/{}_5f_Precision_and_Recall_Score_vs_Thresholds.png".format(method), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # plot of 5g. True Positive Rate vs False Positive Rate (ROC curve)
    fpr, tpr, _ = roc_curve(y_true, prediction_probabilites)
    roc_score = roc_auc_score(y_true, prediction_probabilites, 'macro')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.3f})'.format(roc_score))
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('True Positive Rate vs False Positive Rate (ROC curve) [5g]')
    plt.savefig("./logs/{}_5g_True_Positive_Rate_vs_False_Positive_Rate.png".format(method), bbox_inches='tight', pad_inches=0)
    plt.show()  
    
    log_dict = {
        'all_train_loss': all_train_loss,
        'all_validation_loss': all_validation_loss,
        'all_gradients_norm': all_gradients_norm,
        'iterations': iterations,
        'all_thresholds': all_thresholds,
        'all_precisions': all_precisions,
        'all_recalls': all_recalls,
        'fpr': fpr,
        'tpr': tpr,
        'roc_score': roc_score,
        }
    
    np.save('./logs/{}_log_dict_1.npy'.format(method), log_dict)
    
def check_regularization_effect(add_x0=True, iterations=5000, learning_rate=0.5, regularization_coefficient=0.001, method='non_dp',
          epsilon=0.01, delta=1e-5, data_norm=False, norm_order=2):
    
    regularization_coefficients=[10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    train_losses=[]; val_losses=[];
    X_train, y_train, X_validation, y_validation, X_test, y_test = get_data(data_norm, norm_order)
    for regularization_coefficient in regularization_coefficients:
        print('Current regularization_coefficient: {}'.format(regularization_coefficient))
        best_coefficients, all_train_loss, all_validation_loss, all_gradients_norm, _ = logistic_regression(
                                                                                            X_train, y_train, 
                                                                                            X_validation, y_validation, 
                                                                                            add_x0,
                                                                                            iterations, 
                                                                                            learning_rate,                                                                                                        
                                                                                            regularization_coefficient,
                                                                                            method,
                                                                                            epsilon,
                                                                                            delta,
                                                                                            )
        
        train_losses.append(np.min(all_train_loss))
        val_losses.append(np.min(all_validation_loss))
        time.sleep(1)
        
    all_train_losses = np.array(train_losses)
    all_val_losses = np.array(val_losses)
    regularization_coefficients = np.array(regularization_coefficients)
    
    idx = np.argmin(all_val_losses)
    print('Best val_loss achieved is: {} at regulariztion coefficient: {}'.format(all_val_losses[idx], regularization_coefficients[idx]))
    
    os.makedirs('./logs/', exist_ok=True)
    # plot of 5d. train and validation loss vs regularization coefficients
    plt.plot(regularization_coefficients, all_train_losses, label='train_loss')
    plt.plot(regularization_coefficients, all_val_losses, label='validation_loss')
    plt.xscale('log')
    plt.legend()
    plt.xlabel('regularization coefficients')
    plt.ylabel('loss')
    plt.title('loss vs regularization coefficients [5d]')
    plt.savefig("./logs/{}_5d_loss_vs_regularization_coefficients.png".format(method), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    log_dict = {
        'all_train_loss': all_train_losses,
        'all_validation_loss': all_val_losses,
        'regularization_coefficients': regularization_coefficients,
        }
    
    np.save('./logs/{}_log_dict_2.npy'.format(method), log_dict)

def check_num_samples_effect(add_x0=True, iterations=5000, learning_rate=0.5, regularization_coefficient=0.001, method='non_dp',
          epsilon=0.01, delta=1e-5, data_norm=False, norm_order=2):
    
    keep_train_percentage = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    train_losses=[]; val_losses=[]; all_number_training_samples=[];
    for percentage in keep_train_percentage:
        print('Current percentage: {}'.format(percentage))
        X_train, y_train, X_validation, y_validation, X_test, y_test = get_data(data_norm, norm_order)
        new_len = int(X_train.shape[0]*percentage)
        X_train = X_train[0:new_len,...]
        y_train = y_train[0:new_len]        
        best_coefficients, all_train_loss, all_validation_loss, all_gradients_norm, _ = logistic_regression(
                                                                                            X_train, y_train, 
                                                                                            X_validation, y_validation, 
                                                                                            add_x0,
                                                                                            iterations, 
                                                                                            learning_rate,                                                                                                        
                                                                                            regularization_coefficient,
                                                                                            method,
                                                                                            epsilon,
                                                                                            delta,
                                                                                            )
        
        train_losses.append(np.min(all_train_loss))
        val_losses.append(np.min(all_validation_loss))
        all_number_training_samples.append(new_len)
        time.sleep(1)
        
    all_train_losses = np.array(train_losses)
    all_val_losses = np.array(val_losses)
    all_number_training_samples = np.array(all_number_training_samples)
    
    idx = np.argmin(all_val_losses)
    print('Best val_loss achieved is: {} at num_samples: {}'.format(all_val_losses[idx], all_number_training_samples[idx]))
    
    os.makedirs('./logs/', exist_ok=True)
    # plot of 5d. train and validation loss vs number of training samples
    plt.plot(all_number_training_samples, all_train_losses, label='train_loss')
    plt.plot(all_number_training_samples, all_val_losses, label='validation_loss')
    plt.legend()
    plt.xlabel('number of training samples')
    plt.ylabel('loss')
    plt.title('loss vs number of training samples [5c]')
    plt.savefig("./logs/{}_5c_loss_vs_number_of_training_samples.png".format(method), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    log_dict = {
        'all_train_loss': all_train_losses,
        'all_validation_loss': all_val_losses,
        'all_number_training_samples': all_number_training_samples,
        }
    
    np.save('./logs/{}_log_dict_3.npy'.format(method), log_dict)

def check_step_size_effect(add_x0=True, iterations=5000, learning_rate=0.5, regularization_coefficient=0.001, method='non_dp',
          epsilon=0.01, delta=1e-5, data_norm=False, norm_order=2):
    
    learning_rates=[1, 0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001,
                    0.00075, 0.0005, 0.00025, 0.0001, 0.00005]
    val_losses=[]; all_required_number_iterations=[];
    X_train, y_train, X_validation, y_validation, X_test, y_test = get_data(data_norm, norm_order)
    for learning_rate in learning_rates:
        print('Current step size: {}'.format(learning_rate))
        _, _, all_validation_loss, _, best_iteration = logistic_regression(
                                                                X_train, y_train, 
                                                                X_validation, y_validation, 
                                                                add_x0,
                                                                iterations, 
                                                                learning_rate,                                                                                                        
                                                                regularization_coefficient,
                                                                method,
                                                                epsilon,
                                                                delta,
                                                                )
        
        all_required_number_iterations.append(best_iteration)
        val_losses.append(np.min(all_validation_loss))
        time.sleep(1)
        
    all_required_number_iterations = np.array(all_required_number_iterations)
    all_best_loss = np.array(val_losses)
    step_sizes = np.array(learning_rates)
    
    idx = np.argmin(all_best_loss)
    print('Best val_loss achieved is: {} at step size: {}'.format(all_best_loss[idx], step_sizes[idx]))
    
    os.makedirs('./logs/', exist_ok=True)
    # plot of 5e. Required number of iterations vs Step size
    plt.scatter(step_sizes, all_required_number_iterations, s=80, c=all_best_loss, cmap='jet')
    plt.plot(step_sizes, all_required_number_iterations)
    plt.xscale('log')
    plt.colorbar(label='Best Validation Loss')
    plt.xlabel('Step size')
    plt.ylabel('Required number of iterations')
    plt.title('Required number of iterations vs Step size [5e]')
    plt.savefig("./logs/{}_5e_Required_number_of_iterations_vs_Step_size.png".format(method), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    log_dict = {
        'all_required_number_iterations': all_required_number_iterations,
        'all_best_loss': all_best_loss,
        'step_sizes': step_sizes,
        }
    
    np.save('./logs/{}_log_dict_4.npy'.format(method), log_dict)

def check_acc_roc_vs_epsilon_effect(add_x0=True, iterations=5000, learning_rate=0.5, regularization_coefficient=0.001, method='non_dp',
          epsilon=0.01, delta=1e-5, data_norm=False, norm_order=2):
    
    all_epsilons = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    X_train, y_train, X_validation, y_validation, X_test, y_test = get_data(data_norm, norm_order)
    if add_x0:
        X_test = np.hstack((np.ones((X_test.shape[0], 1)),X_test))
        
    all_acc=[];all_roc=[];
    for epsilon in all_epsilons:
        best_coefficients, _, _, _, _ = logistic_regression(
                                                X_train, y_train, 
                                                X_validation, y_validation, 
                                                add_x0,
                                                iterations, 
                                                learning_rate,                                                                                                        
                                                regularization_coefficient,
                                                method,
                                                epsilon,
                                                delta,
                                                )
        
        logits = np.dot(X_test, best_coefficients)
        prediction_probabilites = sigmoid(logits)
        roc_score = roc_auc_score(y_test, prediction_probabilites, 'macro')
        y_pred = prediction_probabilites >= 0.5
        acc = accuracy_score(y_test, y_pred)
        all_acc.append(acc)
        all_roc.append(roc_score)
    
    os.makedirs('./logs/', exist_ok=True)
    plt.plot(all_epsilons, all_acc)
    plt.xscale('log')
    plt.xlabel('epsilon')
    plt.ylabel('classification accuracy')
    plt.title('classification accuracy vs epsilons for sample size={} and delta=1e-5 [6]'.format(int(X_train.shape[0])))
    plt.savefig("./logs/{}_6_classification_accuracy_vs_epsilon.png".format(method), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    plt.plot(all_epsilons, all_roc)
    plt.xscale('log')
    plt.xlabel('epsilon')
    plt.ylabel('ROC')
    plt.title('ROC vs epsilons for sample size={} and delta=1e-5 [6]'.format(int(X_train.shape[0])))
    plt.savefig("./logs/{}_6_ROC_vs_epsilon.png".format(method), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    log_dict = {
        'all_acc': np.array(all_acc),
        'all_epsilons': np.array(all_epsilons),
        'all_roc': np.array(all_roc)
        }
    
    np.save('./logs/{}_log_dict_5.npy'.format(method), log_dict)

def check_acc_roc_vs_num_samples_effect(add_x0=True, iterations=5000, learning_rate=0.5, regularization_coefficient=0.001, 
                                    method='non_dp', epsilon=0.01, delta=1e-5, data_norm=False, norm_order=2):
    
    keep_train_percentage = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    all_number_training_samples=[];all_acc=[];all_roc=[];
    for percentage in keep_train_percentage:
        print('Current percentage: {}'.format(percentage))
        X_train, y_train, X_validation, y_validation, X_test, y_test = get_data(data_norm, norm_order)
        new_len = int(X_train.shape[0]*percentage)
        X_train = X_train[0:new_len,...]
        y_train = y_train[0:new_len]        
        best_coefficients, _, _, _, _ = logistic_regression(
                                                X_train, y_train, 
                                                X_validation, y_validation, 
                                                add_x0,
                                                iterations, 
                                                learning_rate,                                                                                                        
                                                regularization_coefficient,
                                                method,
                                                epsilon,
                                                delta,
                                                )
        
        if add_x0:
            X_test = np.hstack((np.ones((X_test.shape[0], 1)),X_test))
        logits = np.dot(X_test, best_coefficients)
        prediction_probabilites = sigmoid(logits)
        roc_score = roc_auc_score(y_test, prediction_probabilites, 'macro')
        y_pred = prediction_probabilites >= 0.5
        acc = accuracy_score(y_test, y_pred)
        
        all_acc.append(acc)
        all_roc.append(roc_score)
        all_number_training_samples.append(new_len)
        time.sleep(1)
        
    all_acc = np.array(all_acc)
    all_roc = np.array(all_roc)
    all_number_training_samples = np.array(all_number_training_samples)
    
    idx = np.argmax(all_acc)
    print('Best acc achieved is: {} at num_samples: {}'.format(all_acc[idx], all_number_training_samples[idx]))
    
    idx = np.argmax(all_roc)
    print('Best roc achieved is: {} at num_samples: {}'.format(all_roc[idx], all_number_training_samples[idx]))
    
    os.makedirs('./logs/', exist_ok=True)
    # plot of 7 accuracy vs number of training samples
    plt.plot(all_number_training_samples, all_acc)
    plt.xlabel('number of training samples')
    plt.ylabel('classification accuracy')
    plt.title('classification accuracy vs number of training samples [7]')
    plt.savefig("./logs/{}_7_classification_accuracy_vs_number_of_training_samples.png".format(method), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    os.makedirs('./logs/', exist_ok=True)
    # plot of 7 ROC vs number of training samples
    plt.plot(all_number_training_samples, all_roc)
    plt.xlabel('number of training samples')
    plt.ylabel('ROC')
    plt.title('ROC vs number of training samples [7]')
    plt.savefig("./logs/{}_7_roc_vs_number_of_training_samples.png".format(method), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    log_dict = {
        'all_acc': all_acc,
        'all_roc': all_roc,
        'all_number_training_samples': all_number_training_samples,
        }
    
    np.save('./logs/{}_log_dict_6.npy'.format(method), log_dict)
   
def plot_mnist_examples():
    X_train, y_train, X_validation, y_validation, X_test, y_test = get_data()
    a = X_train[0].reshape([28,28])
    b = X_train[1].reshape([28,28])
    c = X_train[2].reshape([28,28])
    d = X_train[-1].reshape([28,28])
    plt.subplot(1, 4, 1);plt.imshow(a, cmap='gray');plt.axis('off');
    plt.subplot(1, 4, 2);plt.imshow(b, cmap='gray');plt.axis('off');
    plt.subplot(1, 4, 3);plt.imshow(c, cmap='gray');plt.axis('off');
    plt.subplot(1, 4, 4);plt.imshow(d, cmap='gray');plt.axis('off');
    os.makedirs('./logs/', exist_ok=True)
    plt.savefig("./logs/mnist_examples.png", bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    # fig a,b,f,g for non_dp, gaussian, laplace
    train(iterations=5000, method='non_dp', data_norm=False)
    train(iterations=5000, method='gaussian', data_norm=True, norm_order=2)
    train(iterations=5000, learning_rate=5, regularization_coefficient=0.001, method='laplace', data_norm=True, norm_order=1) 
    
    # fig d for non_dp, gaussian, laplace
    check_regularization_effect(iterations=5000, method='non_dp', data_norm=False)
    check_regularization_effect(iterations=5000, method='gaussian', data_norm=True, norm_order=2)
    check_regularization_effect(iterations=5000, learning_rate=5, regularization_coefficient=0.001, method='laplace', data_norm=True, norm_order=1) 
    
    # fig c for non_dp, gaussian, laplace
    check_num_samples_effect(iterations=5000, method='non_dp', data_norm=False)
    check_num_samples_effect(iterations=5000, method='gaussian', data_norm=True, norm_order=2)
    check_num_samples_effect(iterations=5000, learning_rate=5, regularization_coefficient=0.001, method='laplace', data_norm=True, norm_order=1) 
    
    # fig e for non_dp, gaussian, laplace
    check_step_size_effect(iterations=5000, method='non_dp', data_norm=False)
    check_step_size_effect(iterations=5000, method='gaussian', data_norm=True, norm_order=2)
    check_step_size_effect(iterations=5000, learning_rate=5, regularization_coefficient=0.001, method='laplace', data_norm=True, norm_order=1)
    
    # fig 6 for non_dp, gaussian, laplace
    check_acc_roc_vs_epsilon_effect(iterations=5000, method='non_dp', data_norm=False)
    check_acc_roc_vs_epsilon_effect(iterations=5000, method='gaussian', data_norm=True, norm_order=2)
    check_acc_roc_vs_epsilon_effect(iterations=5000, learning_rate=5, regularization_coefficient=0.001, method='laplace', data_norm=True, norm_order=1)
    
    # fig 7 for non_dp, gaussian, laplace
    # check_acc_roc_vs_num_samples_effect(iterations=5000, method='non_dp', data_norm=False)
    # check_acc_roc_vs_num_samples_effect(iterations=5000, method='gaussian', data_norm=True, norm_order=2)
    # check_acc_roc_vs_num_samples_effect(iterations=5000, learning_rate=5, regularization_coefficient=0.001, method='laplace', data_norm=True, norm_order=1) 
    
    # plot mnist example
    # plot_mnist_examples()