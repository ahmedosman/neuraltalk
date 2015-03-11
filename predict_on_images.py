import argparse
import json
import time
import datetime
import numpy as np
import code
import os
import cPickle as pickle
import math
import scipy.io

from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split
import caffe

"""
This script is used to predict sentences for arbitrary images
that are located in a folder we call root_folder. It is assumed that
the root_folder contains:
- the raw images
- a file tasks.txt that lists the images you'd like to use
- a file vgg_feats.mat that contains the CNN features. 
  You'll need to use the Matlab script I provided and point it at the
  root folder and its tasks.txt file to save the features.

Then point this script at the folder and at a checkpoint model you'd
like to evaluate.
"""


def extract_feats(path_imgs , path_model_def , path_model , batch_size = 1 , WITH_GPU = True):
    '''
    Function using the caffe python wrapper to extract 4096 from VGG_ILSVRC_16_layers.caffemodel model
    
    Inputs:
    ------
    path_imgs      : list of the full path of images to be processed 
    path_model_def : path to the model definition file
    path_model     : path to the pretrained model weight
    WItH_GPU       : Use a GPU 
    
    Output:
    -------
    features           : return the features extracted 
    '''
    
    if WITH_GPU:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    
    caffe_net = caffe.Classifier(path_model_def , path_model , image_dims = (224,224) , raw_scale = 255,
                            mean = np.array([103.939, 116.779, 123.68]) )

    feats = np.zeros((4096 , len(path_imgs)))
    
    for b in range(0 , len(path_imgs) , batch_size):
        list_imgs = []
        for i in range(b , b + batch_size ):
            if i < len(path_imgs):
                list_imgs.append( np.array(caffe.io.load_image(path_imgs[i]) ) )
            else:
                list_imgs.append(list_imgs[-1]) #Appending the last image in order to have a batch of size 10. The extra predictions are removed later..
                
        caffe_input = np.asarray([caffe_net.transformer.preprocess('data', in_) for in_ in list_imgs]) #preprocess the images
       
        predictions =caffe_net.forward(data = caffe_input)
        predictions = predictions[caffe_net.outputs[0]].transpose()
        
        if i < len(path_imgs):
            feats[:,b:i+1] = predictions
            n = i+1
        else:
            n = min(batch_size , len(path_imgs) - b) 
            feats[:,b:b+n] = predictions[:,0:n] #Removing extra predictions, due to the extra last image appending.
            n += b 
        print "%d out of %d done....."%(n ,len(path_imgs))

    return feats
        
    
            
        
def main(params):

  # load the checkpoint
  checkpoint_path = params['checkpoint_path']
  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']
  dataset = checkpoint_params['dataset']
  model = checkpoint['model']
  misc = {}
  misc['wordtoix'] = checkpoint['wordtoix']
  ixtoword = checkpoint['ixtoword']

  # output blob which we will dump to JSON for visualizing the results
  blob = {} 
  blob['params'] = params
  blob['checkpoint_params'] = checkpoint_params
  blob['imgblobs'] = []

  # load the tasks.txt file
  root_path = params['root_path']
  img_names = open(os.path.join(root_path, 'tasks.txt'), 'r').read().splitlines()

  # load the features for all images
  features_path = os.path.join(root_path, 'vgg_feats.mat')
  
  if not os.path.exists(features_path):
      print "Generating features for the images on disk"
      path_imgs = [ os.path.join(root_path , img) for img in img_names]
      path_model_def = '/home/ahmedosman/Documents/caffe_model/VGG/deploy_features.prototxt'
      path_model = '/home/ahmedosman/Documents/caffe_model/VGG/VGG_ILSVRC_16_layers.caffemodel'
      features = extract_feats(path_imgs, path_model_def, path_model, batch_size = 10, WITH_GPU = False)
      feature_path = os.path.join(root_path , 'vgg_feats.mat')
      print "Saving features to disk %s"%(feature_path)
      features_dict = {}
      features_dict['feats'] = features
      scipy.io.savemat(feature_path, features_dict)
      
       
  else:
      features_struct = scipy.io.loadmat(features_path)
      features = features_struct['feats'] # this is a 4096 x N numpy array of features
  D,N = features.shape

  # iterate over all images and predict sentences
  BatchGenerator = decodeGenerator(checkpoint_params)
  for n in xrange(N):
    print 'image %d/%d:' % (n, N)

    # encode the image
    img = {}
    img['feat'] = features[:, n]
    img['local_file_path'] =img_names[n]

    # perform the work. heavy lifting happens inside
    kwparams = { 'beam_size' : params['beam_size'] }
    Ys = BatchGenerator.predict([{'image':img}], model, checkpoint_params, **kwparams)

    # build up the output
    img_blob = {}
    img_blob['img_path'] = img['local_file_path']

    # encode the top prediction
    top_predictions = Ys[0] # take predictions for the first (and only) image we passed in
    top_prediction = top_predictions[0] # these are sorted with highest on top
    candidate = ' '.join([ixtoword[ix] for ix in top_prediction[1] if ix > 0]) # ix 0 is the END token, skip that
    print 'PRED: (%f) %s' % (top_prediction[0], candidate)
    img_blob['candidate'] = {'text': candidate, 'logprob': top_prediction[0]}    
    blob['imgblobs'].append(img_blob)

  # dump result struct to file
  save_file = os.path.join(root_path, 'result_struct.json')
  print 'writing predictions to %s...' % (save_file, )
  json.dump(blob, open(save_file, 'w'))

  # dump output html
  html = ''
  for img in blob['imgblobs']:
    html += '<img src="%s" height="400"><br>' % (img['img_path'], )
    html += '(%f) %s <br><br>' % (img['candidate']['logprob'], img['candidate']['text'])
  html_file = os.path.join(root_path, 'result.html')
  print 'writing html result file to %s...' % (html_file, )
  open(html_file, 'w').write(html)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('-r', '--root_path', default='example_images', type=str, help='folder with the images, tasks.txt file, and corresponding vgg_feats.mat file')
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
