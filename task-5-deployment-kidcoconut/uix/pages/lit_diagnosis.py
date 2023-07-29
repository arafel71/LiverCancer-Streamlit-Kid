#--- anomaly detection - supervised page
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms.functional import to_tensor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import lib.utils as libUtils

import sys
import os, random, io

description = "Diagnosis"
m_kblnTraceOn = True                                  #--- enable/disable module level tracing


#--- model initializations
#data_batch_size = 3         #--- decrease the number of images loaded, processed if the notebook crashes due to limited RAM
#NUM_EPOCHS = 10 # 50
#BATCH_SIZE = data_batch_size
NUM_CLASSES = 3

# path to save model weights
#BESTMODEL_PATH = r"model_deeplabv3_r50_full_training_dataset_80-20_split_10epochs_no-norm_vhflip30.pth"  #--- path to save model weights
BESTMODEL_PATH = r"model.pth"
MODEL_FULLPATH = 'bin/models/' + BESTMODEL_PATH
model_path = MODEL_FULLPATH

DEFAULT_DEVICE_TYPE = ('cuda' if torch.cuda.is_available() else 'cpu')       #--- cuda if gpu; cpu if on Colab Free
DEFAULT_BACKBONE_MODEL = 'r50'
backbone_model_name = DEFAULT_BACKBONE_MODEL



def image_toBytesIO(image: Image) -> bytes:
  #--- BytesIO is a file-like buffer stored in memory
  imgByteArr = io.BytesIO()

  #--- image.save expects a file-like as a argument
  image.save(imgByteArr, format=image.format)

  return imgByteArr


def image_toByteArray(image: Image) -> bytes:
  #--- convert image to bytesIO 
  imgByteArr = image_toBytesIO(image)

  #--- Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


def run():
    #--- note:  in python, you need to specify global scope for fxns to access module-level variables 
    global m_kbln_TraceOn
    print("\nINFO (litDiagnosis.run)  loading ", description, " page ...") 


    #--- page settings
    if (m_kblnTraceOn):  print("TRACE1 (litDiagnosis.run):  Initialize Page Settings ...")
    #st.header("Single Tile Diagnosis")
    st.markdown("#### Single Tile Diagnosis")

    #--- allow the user to select a random sample
    imgUploaded = None
    if st.button("Random Sample"):
        #--- get a random sample file
        strPth_sample = libUtils.pth_dtaTileSamples
        strFil_sample = random.choice(os.listdir(strPth_sample))
        strFullPth_sample = os.path.join(strPth_sample, strFil_sample)

        print("INFO (litDiagnosis.run):  sample file selected ... ", strFullPth_sample)

        #--- display;  convert file image to bytesIO
        imgSample = Image.open(strFullPth_sample)
        imgSample = image_toBytesIO(imgSample)
        imgUploaded = imgSample
        imgUploaded.name = strFil_sample
        imgUploaded.type = os.path.splitext(strFil_sample)[1]


    #--- provide file drag/drop capability
    m_blnDisableDragDrop = False
    #if(not m_blnDisableDragDrop): 
        #btnSave = st.button("Save")
    imgDropped = st.file_uploader("Upload a single Tile", 
                type=["png", "jpg", "tif", "tiff", "img"], 
                accept_multiple_files=False )
    #m_blnDisableDragDrop = (imgDropped is None)
    #--- <class 'streamlit.runtime.uploaded_file_manager.UploadedFile'>
    if (imgDropped is not None):
        imgUploaded = imgDropped


    if (imgUploaded is None):
        if (m_kblnTraceOn):  print("ERROR (litDiagnosis.run):  imgUploaded is None ...")
    else:
        try:
            #--- display uploaded file details
            if (m_kblnTraceOn):  print("TRACE1 (litDiagnosis.run):  Print uploaded file details ...")
            st.write("FileName:", "&nbsp;&ensp;&emsp;", imgUploaded.name)
            st.write("FileType:", "&nbsp;&emsp;&emsp;", imgUploaded.type)
            
            #--- show:  
            #if (m_kblnTraceOn):  print("TRACE (litDiagnosis.run):  load WSI ...")
            #if (m_blnDisableDragDrop):
                #--- load wsi 
            #    print("")
            #else:

                #--- display diagnosis results ... format (vertical)
                #showDiagnosis_vert(imgUploaded)
            showDiagnosis_horiz(imgUploaded)

        except TypeError as e:
            print("ERROR (litDiagnosis.run_typeError1):  ", e)

        except:
            e = sys.exc_info()
            print("ERROR (litDiagnosis.run_genError1):  ", e)  


        try:

            #--- display WSI
            #showImg_wsi(img)
            #st.image("bin/images/sample_wsi.png", use_column_width=True)

            print("")

        except TypeError as e:
            print("ERROR (litDiagnosis.run_typeError2):  ", e)

        except:
            e = sys.exc_info()
            print("ERROR (litDiagnosis.run_genError2):  ", e)  


def showImg_wsi(img):
     print("")


def readyModel_getPreds(imgDropped):
    print("TRACE:  readyModel_getPreds ...")
    print("INFO:  save raw tile ...")
    strPth_tilRaw = save_tilRaw(imgDropped)
    
    #--- ready the model
    print("INFO:  ready base model ...")
    mdlBase = readyBaseModel()
    print("INFO:  ready model with weights ...")
    mdlWeights = readyModelWithWeights(mdlBase)
    print("INFO:  ready model with xai ...")
    mdlXai = readyModelWithXAI(mdlWeights)

    #--- get the XAI weighted prediction
    print("INFO:  get xai weighted pred ...")
    output_pred, tns_batch = predXai_tile(mdlXai, strPth_tilRaw)

    #--- get the GRADCAM predictions
    print("INFO:  get GRADCAM preds ...")
    cam_img_bg, cam_img_wt, cam_img_vt = predGradCam_tile(output_pred, mdlXai, tns_batch)

    print("TRACE:  return readyModel_getPreds ...")
    return strPth_tilRaw, output_pred, cam_img_bg, cam_img_wt, cam_img_vt


def showDiagnosis_horiz(imgDropped):

    #--- copy the uploaded file to data/tiles/raw       
    st.write("#")

    #--- ready the model, get predictions
    print("TRACE2:  ready model ...")
    strPth_tilRaw, xai_pred, cam_img_bg, cam_img_wt, cam_img_vt = readyModel_getPreds(imgDropped)

    #--- display the raw prediction:  headers
    print("TRACE2:  display raw preds, headers ...")
    colRaw, colPred, colGradBack, colGradWhole, colGradViable = st.columns(5)
    colRaw.write("Raw Tile")
    colPred.write("Prediction")
    colGradBack.write("GradCAM:  Background")
    colGradWhole.write("GradCAM:  Whole Tumor")
    colGradViable.write("GradCAM:  Viable Tumor")

    #--- display the raw prediction:  images
    colRaw, colPred, colGradBack, colGradWhole, colGradViable = st.columns(5)
    showCol_rawTil(colRaw, strPth_tilRaw)
    showCol_predTil(colPred, xai_pred[0], strPth_tilRaw)
    showCol_gradCamImg("imgGradCam_bg", colGradBack, cam_img_bg[0])
    showCol_gradCamImg("imgGradCam_wt", colGradWhole, cam_img_wt[0])
    showCol_gradCamImg("imgGradCam_vt", colGradViable, cam_img_vt[0])


def showCol_rawTil(colRaw, strPth_tilRaw):
    print("TRACE3:  showCol_rawTil ...")
    colRaw.image(strPth_tilRaw, width=400, use_column_width=True)

#--- Dark blue -> Background
#   Brown -> Whole tumor
#   Green/Aqua -> Viable tumor
def showCol_predTil(colPred, xai_pred, strPth_tilRaw):
    kstrPth_tilePred = "data/tiles/pred/"
    strFilName = os.path.basename(strPth_tilRaw)
    strFil_tilePred = kstrPth_tilePred + strFilName
    
    #--- make sure the dir exists
    ensureDirExists(kstrPth_tilePred)

    print("TRACE3:  showCol_predTil2 ... ", strFil_tilePred)
    argmax_mask = torch.argmax(xai_pred, dim=0)
    preds = argmax_mask.cpu().squeeze().numpy()

    cmap = plt.cm.get_cmap('tab10', 3)  # Choose a colormap with 3 colors
    print("TRACE3:  typeOf(preds) ...", type(preds))

    print("TRACE3:  save pred image ...")
    plt.imsave(strFil_tilePred, preds, cmap=cmap, vmin=0, vmax=2)

    print("TRACE3:  load image ...", strFil_tilePred)
    colPred.image(strFil_tilePred, width=400, use_column_width=True)


def showCol_gradCamImg(strImgContext, colGradCam, cam_img):
    print("TRACE3:  showCol_gradImg ... ", strImgContext)
    imgGradCam = Image.fromarray(cam_img)
    colGradCam.image(imgGradCam, width=400, use_column_width=True)


def showDiagnosis_vert(imgDropped):

    #--- copy the uploaded file to data/tiles/raw       
    st.write("#")

    #--- ready the model, get predictions
    strPth_tilRaw, xai_pred, cam_img_bg, cam_img_wt, cam_img_vt = readyModel_getPreds(imgDropped)

    #--- display all predictions
    '''
    strPth_tilPred = save_tilPred(output_pred)
    strPth_tilGradBg = save_tilGradBg(cam_img_bg)
    strPth_tilGradWt = None
    strPth_tilGradVt = None
    '''

    #--- display the raw image
    lstDescr = ["Raw Tile", "Prediction", "GradCam: Background", "GradCam: Whole Tumor", "GradCam: Viable Tumor"]
    lstImages = [strPth_tilRaw, strPth_tilRaw, strPth_tilRaw, strPth_tilRaw, strPth_tilRaw]

    #--- display the raw prediction 
    for imgIdx in range(len(lstImages)):
        colDescr, colImage = st.columns([0.25, 0.75])
        colDescr.write(lstDescr[imgIdx])
        colImage.image(lstImages[imgIdx], width=400, use_column_width=True)


def ensureDirExists(strPth):
    blnExists = os.path.exists(strPth)
    if not blnExists:
        os.makedirs(strPth)
    print("TRACE:  creating dir ... ", strPth)


def save_tilRaw(imgDropped):
    print("TRACE:  save_tilRaw ...")
    #--- copy the uploaded raw Tile to data/tiles/raw
    kstrPth_tileRaw = "data/tiles/raw/"
    strFil_tileRaw = kstrPth_tileRaw + imgDropped.name
    print("TRACE:  save_tilRaw.file ... ", strFil_tileRaw)

    #--- make sure the dir exists
    ensureDirExists(kstrPth_tileRaw)

    #--- check if the file already exists; delete
    if (os.path.isfile(strFil_tileRaw)):
        print("WARN:  save_tilRaw.file exists;  delete ... ", strFil_tileRaw)
        os.remove(strFil_tileRaw)

    with open(strFil_tileRaw,"wb") as filUpload: 
        filUpload.write(imgDropped.getbuffer()) 
    print("TRACE:  uploaded file saved to ", strFil_tileRaw)        
    return strFil_tileRaw


def prepare_model(backbone_model="mbv3", num_classes=2):

    # Initialize model with pre-trained weights.
    weights = 'DEFAULT'
    if backbone_model == "mbv3":
        model = None
        #model = deeplabv3_mobilenet_v3_large(weights=weights)

    elif backbone_model == "r50":
        model = deeplabv3_resnet50(weights=weights)

    elif backbone_model == "r101":
        model = None
        #model = deeplabv3_resnet101(weights=weights)
        
    else:
        raise ValueError("Wrong backbone model passed. Must be one of 'mbv3', 'r50' and 'r101' ")

    # Update the number of output channels for the output layer.
    # This will remove the pre-trained weights for the last layer.
    model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, num_classes, kernel_size=1)
    model.aux_classifier[-1] = nn.Conv2d(model.aux_classifier[-1].in_channels, num_classes, kernel_size=1)
    return model


# computes IoU or Dice index
def intermediate_metric_calculation(predictions, targets,  use_dice=False,
                                    smooth=1e-6, dims=(2, 3)):
    # dims corresponding to image height and width: [B, C, H, W].

    # Intersection: |G ∩ P|. Shape: (batch_size, num_classes)
    intersection = (predictions * targets).sum(dim=dims) + smooth

    # Summation: |G| + |P|. Shape: (batch_size, num_classes).
    summation = (predictions.sum(dim=dims) + targets.sum(dim=dims)) + smooth

    if use_dice:
        # Dice Shape: (batch_size, num_classes)
        metric = (2.0 * intersection) / summation
    else:
        # Union. Shape: (batch_size, num_classes)
        union = summation - intersection + smooth

        # IoU Shape: (batch_size, num_classes)
        metric = intersection /  union

    # Compute the mean over the remaining axes (batch and classes).
    # Shape: Scalar
    total = metric.mean()

    #print(f"iou = {total}")
    return total


def convert_2_onehot(matrix, num_classes=3):
    '''
    Perform one-hot encoding across the channel dimension.
    '''
    matrix = matrix.permute(0, 2, 3, 1)
    matrix = torch.argmax(matrix, dim=-1)
    matrix = torch.nn.functional.one_hot(matrix, num_classes=num_classes)
    matrix = matrix.permute(0, 3, 1, 2)
    return matrix


#--- I'm using just categorical cross_entropy for now
class Loss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, predictions, targets):
    # predictions --> (B, #C, H, W) unnormalized
    # targets     --> (B, #C, H, W) one-hot encoded
    targets = torch.argmax(targets, dim=1)
    pixel_loss = F.cross_entropy(predictions, targets, reduction="mean")

    return pixel_loss
  
  
class Metric(nn.Module):
    def __init__(self, num_classes=3, smooth=1e-6, use_dice=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.use_dice    = use_dice

    def forward(self, predictions, targets):
        # predictions  --> (B, #C, H, W) unnormalized
        # targets      --> (B, #C, H, W) one-hot encoded

        # Converting unnormalized predictions into one-hot encoded across channels.
        # Shape: (B, #C, H, W)
        predictions = convert_2_onehot(predictions, num_classes=self.num_classes) # one hot encoded
        metric = intermediate_metric_calculation(predictions, targets, use_dice=self.use_dice, smooth=self.smooth)

        # Compute the mean over the remaining axes (batch and classes). Shape: Scalar
        return metric


def get_default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readyBaseModel():

    #--- prep model conditions
    device = get_default_device()
    model = prepare_model(backbone_model=backbone_model_name, num_classes=NUM_CLASSES)

    metric_name = "iou"
    use_dice = (True if metric_name == "dice" else False)
    metric_fn = Metric(num_classes=NUM_CLASSES, use_dice=use_dice).to(device)
    loss_fn   = Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    return model


def readyModelWithWeights(mdlBase):
    print("TRACE:  loading model with weights ... ", model_path)    
    mdlBase.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_with_weights = mdlBase
    model_with_weights.eval()
    return model_with_weights


class SegmentationModelOutputWrapper(torch.nn.Module):
  def __init__(self, model):
      super(SegmentationModelOutputWrapper, self).__init__()
      self.model = model

  def forward(self, x):
    return self.model(x)["out"]
  

def readyModelWithXAI(mdlWeighted):
    model_xai = SegmentationModelOutputWrapper(mdlWeighted)

    model_xai.eval()
    model_xai.to('cpu')
    return model_xai


#--- demo:  process a single file for validation/demo
def val_filToTensor(strPth_fil):
  img_fil = Image.open(strPth_fil)
  img_fil = img_fil.convert("RGB")
  img_fil = np.asarray(img_fil)/255
  return to_tensor(img_fil).unsqueeze(0)


#--- TODO demo:  process a batch of files for validation/demo
def val_aryToTensor(pth_fil, ary_fils):
  aryTensor = []
  for str_filName in ary_fils:
    aryTensor.append(val_filToTensor(pth_fil, str_filName))
  return aryTensor


def predXai_tile(mdl_xai, strPth_tileRaw):
    #--- run a prediction for a single 
    print("TRACE:  get tensor from single file ... ", strPth_tileRaw)
    val_tensorFil = val_filToTensor(strPth_tileRaw)
    val_tensorBatch = val_tensorFil

    print("TRACE:  get mdl_xai prediction ...")
    output = mdl_xai(val_tensorBatch.float().to('cpu'))

    print("TRACE:  predXai_tile return ...")
    return output, val_tensorBatch


class SemanticSegmentationTarget:
  def __init__(self, category, mask):
    self.category = category
    self.mask = torch.from_numpy(mask)
    if torch.cuda.is_available():
        self.mask = self.mask.cuda()

  def __call__(self, model_output):
    return (model_output[self.category, :, : ] * self.mask).sum()
  

def predGradCam_tile(output_xaiPred, mdl_xai, val_image_batch):
    print("TRACE:  predGradCam initialize ...")
    cam_img_bg = []
    cam_img_wt = []
    cam_img_vt = []

    sem_classes = ['__background__', 'whole_tumor', 'viable_tumor']
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    argmax_mask = torch.argmax(output_xaiPred, dim=1)
    argmax_mask_np = argmax_mask.cpu().squeeze().numpy()
    preds = argmax_mask_np

    seg_mask = preds
    bg_category = sem_class_to_idx["__background__"]
    bg_mask_float = np.float32(seg_mask == bg_category)
    wt_category = sem_class_to_idx["whole_tumor"]
    wt_mask_float = np.float32(seg_mask == wt_category)
    vt_category = sem_class_to_idx["viable_tumor"]
    vt_mask_float = np.float32(seg_mask == vt_category)

    target_layers = [mdl_xai.model.backbone.layer4]

    for i in range(len(val_image_batch)):
        rgb_img = np.float32(val_image_batch[i].permute(1, 2, 0))
        rgb_tensor = val_image_batch[i].unsqueeze(0).float()

        print("TRACE:  process the background ...")
        targets = [SemanticSegmentationTarget(bg_category, bg_mask_float[i])]
        with GradCAM(model=mdl_xai,
                    target_layers=target_layers,
                    use_cuda=torch.cuda.is_available()) as cam:

                    grayscale_cam = cam(input_tensor = rgb_tensor,
                                        targets = targets)[0, :]
                    cam_img_bg.append(show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True))

        print("TRACE:  process whole tumors ...")
        targets = [SemanticSegmentationTarget(wt_category, wt_mask_float[i])]
        with GradCAM(model=mdl_xai,
                    target_layers=target_layers,
                    use_cuda=torch.cuda.is_available()) as cam:

                    grayscale_cam = cam(input_tensor = rgb_tensor,
                                        targets = targets)[0, :]
                    cam_img_wt.append(show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True))

        print("TRACE:  process viable tumors ...")
        targets = [SemanticSegmentationTarget(vt_category, vt_mask_float[i])]
        with GradCAM(model=mdl_xai,
                    target_layers=target_layers,
                    use_cuda=torch.cuda.is_available()) as cam:

                    grayscale_cam = cam(input_tensor = rgb_tensor,
                                        targets = targets)[0, :]
                    cam_img_vt.append(show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True))
    
    return cam_img_bg, cam_img_wt, cam_img_vt
