from  utility.Dataset import Dataset as ds
from  utility.CNN import CNN as cnn
import argparse
seed=42


#read arguments

parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', type=str, required=False)
parser.add_argument('--lr', type=float, required=False)
parser.add_argument('--dropout', type=float, required=False)
parser.add_argument('--image_size', type=int, required=False)
parser.add_argument('--batchNormalization', type=bool, required=False)
parser.add_argument('--epoch', type=int, required=False)
parser.add_argument('--filterOrganization', type=str, required=True)
parser.add_argument('--activation_function', type=str, required=False)
parser.add_argument('--number_of_neurons_in_the_dense_layer', type=int, required=False)
parser.add_argument('--augment_data', type=bool, required=False)
parser.add_argument('--wandbLog', type=bool, required=False)
parser.add_argument( '--no_of_filters', help='delimited list input', type=str,required=False)
args = parser.parse_args()


if(args.no_of_filters is None):
    no_of_filters = [64,64,64,64,64]
else:
    no_of_filters = [int(item) for item in args.no_of_filters.split(',')]


if(args.number_of_neurons_in_the_dense_layer is None):
    number_of_neurons_in_the_dense_layer=512
else:
    number_of_neurons_in_the_dense_layer=args.number_of_neurons_in_the_dense_layer

if(args.augment_data is None):
    augment_data=True
else:
    augment_data=args.augment_data

if(args.wandbLog is None):
    wandbLog=False
else:
    wandbLog=args.wandbLog
if(args.filterOrganization is None):
    filterOrganization='config_all_same'
else:
    filterOrganization=args.filterOrganization

if(args.activation_function is None):
    activation_function='relu'
else:
    activation_function=args.activation_function

if(args.optimizer is None):
    optimizer='adam'
else:
    optimizer=args.optimizer



if(args.epoch is None):
    epoch=5
else:
    epoch=args.epoch

if(args.dropout is None):
    dropout=0.2
else:
    dropout=args.dropout

if(args.lr is None):
    lr=1e-4
else:
    lr=args.lr
if(args.batchNormalization is None):
    batchNormalization=True
else:
    batchNormalization=args.batchNormalization

if(args.image_size is None):
    image_size=224
else:
    image_size=args.image_size

#sample inputs



print('------------------------------------------')
print('Summary of Parameters')
print('------------------------------------------')
print('image_size :'+str(image_size))
print('batchNormalization :'+str(batchNormalization))
print('lr :'+str(lr))
print('dropout :'+str(dropout))
print('epoch :'+str(epoch))
print('optimizer :'+str(optimizer))
print('activation_function :'+str(activation_function))
valid_filterOrganization=['config_all_same','config_incr','config_decr','config_alt_incr','config_alt_decr']
if(filterOrganization  in valid_filterOrganization):
    print('filterOrganization :'+str(filterOrganization))
else:
    print('no_of_filters :'+str(no_of_filters))
print('wandbLog :'+str(wandbLog))
print('number_of_neurons_in_the_dense_layer :'+str(number_of_neurons_in_the_dense_layer))
print('augment_data :'+str(augment_data))


print('------------------------------------------')

ds.downloadDataSet()
train_ds,val_ds,test_ds=ds.import_dataset(seed=42,image_size=image_size,augment_data=augment_data)

no_of_filters = [32,32,32,32,32]
size_of_filters = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]

number_of_classes=10
model=cnn.setUp(no_of_filters,size_of_filters,\
          activation_function,\
          number_of_neurons_in_the_dense_layer,\
          number_of_classes,\
          dropout,batchNormalization,\
          filterSize=16,\
         filterOrganization=filterOrganization,imsize=image_size) 



model=cnn.train(model,train_ds,val_ds,optimizer,lr,epoch,wandbLog)
model.evaluate(test_ds)
    
 
