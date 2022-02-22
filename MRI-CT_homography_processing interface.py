import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import pandas as pd
import cv2
import matplotlib.cm as cm
from skimage import filters
from skimage.measure import regionprops, label
from os.path import exists
import pickle
#%%

def change_middle_plane(images, crossection, n):
    images_changed = np.zeros([images.shape[0],images.shape[1],images.shape[2]])
    
    diff = abs(crossection)-round((n+0.1)/2)
    if diff == 0:
        images_changed = images
    elif diff>0:
        images_changed[0:(n-diff)]= images[diff:n]
        images_changed[(n-diff):n] = images[0:diff]
    else: 
        images_changed[0:(-diff)] = images[diff:n]
        images_changed[(-diff):n] = images[0:(n+diff)]
    
    return images_changed

class MyFigureCanvas(object):
    def __init__(self, fig, participant,ct, mri, z, path):
        
        # init class attributes:
        self.path = path
        self.figure = fig
        self.participant = participant
        self.z = z
        self.ct = ct
        self.mri = mri
        self.msize = 5
        self.ct_x = 0.0
        self.ct_y = 0.0
        self.mri_x = 0.0
        self.mri_y = 0.0
        self.ct_data = np.zeros([8,2,z*2])
        self.mri_data = np.zeros([8,2,z*2])
        self.weighted_center_of_mass_middle_ct = 0.0
        self.weighted_center_of_mass_middle_mri = 0.0
        self.area_middle_mri = 0.0
        self.area_middle_ct = 0.0
        self.alpha = 0.5
        
        self.background_mri = None
        self.background_ct = None
        self.background_homl = None
        self.background_homr = None
        self.draggable_mri = None
        self.draggable_ct = None
        
        self.homography_image = np.zeros(self.mri.shape)
        self.transformations = []
        for i in range(len(self.mri)):
            self.transformations.append([])
        
        self.slices_mri, self.rows_mri, self.cols_mri = self.mri.shape
        self.slices_ct, self.rows_ct, self.cols_ct = self.ct.shape
        self.ind = z
        self.suptitle = self.figure.suptitle('')
        
        self.flag_arrow = False
        self.flag_homo = False
        self.flag_markers = 0
        self.flag_propagation = np.zeros(z*2)

        # showing the  middle images
        self.figure.suptitle(self.participant)
        self.grid = plt.GridSpec(2, 2, wspace = 0.25, hspace = 0.25)
        self.ax = plt.subplot(self.grid[0])
        # self.ax = self.figure.add_subplot(221)
        self.im_mri = self.ax.imshow(self.mri[self.ind])
        self.ax.set_title('MRI')
        self.ax.set_ylabel(str(self.ind))
        
        self.ax2 = plt.subplot(self.grid[1], sharex = self.ax, sharey = self.ax)
        self.im_ct = self.ax2.imshow(self.ct[self.ind])
        self.ax2.set_title('CT')
        self.ax2.set_ylabel(str(self.ind))
        
        self.ax3 = plt.subplot(self.grid[2], sharex = self.ax, sharey = self.ax)
        self.ax3.set_title('MRI - CT')
        self.ax4 = plt.subplot(self.grid[3], sharex = self.ax, sharey = self.ax)
        self.ax4.set_title('CT - MRI')
        
        
        
        # checking if the file exists to enables if there are markers or it should be marked
        self.file_exists_mri = exists(self.path+'_selected points'+'_mri.npy')
        self.file_exists_ct = exists(self.path+'_selected points'+'_ct.npy')
        
        if self.file_exists_mri and self.file_exists_ct:
            self.mri_data = np.load(self.path+'_selected points'+'_mri.npy')
            self.ct_data = np.load(self.path+'_selected points'+'_ct.npy')
            self.homography_image = np.load(self.path+'_new_ct.npy')
            with open(self.path + '_transformations.data', "rb") as filehandle:
                self.transformations = pickle.load(filehandle)
            self.mri_x = self.mri_data[:,0,self.ind] 
            self.mri_y = self.mri_data[:,1,self.ind]
            self.ct_x = self.ct_data[:,0,self.ind] 
            self.ct_y = self.ct_data[:,1,self.ind]
            self.flag_markers = 1
            self.flag_propagation = np.load(self.path+'_flag_propagation.npy')
            self.mi = self.mutual_information()
            text = 'MI = '+ str(round(self.mi,2))
            # self.figure.suptitle(self.participant +'\n'+text)
            self.suptitle.set_text(self.participant +'\n'+text)
            
            
        else:
            self.selected_points = self.figure.ginput(16,timeout= 0)
            self.selected_points = np.asarray(self.selected_points)
            self.x_sel = self.selected_points[:,0]
            self.y_sel = self.selected_points[:,1]
            self.ct_x = self.x_sel[0::2]
            self.ct_y = self.y_sel[0::2]
            self.mri_x = self.x_sel[1::2]
            self.mri_y = self.y_sel[1::2]
            self.flag_markers = 2
            print(self.ind)
        
        if self.flag_markers == 2:
            self.colors = cm.rainbow(np.linspace(0, 1, len(self.ct_y)))
            self.ax.scatter(self.mri_x, self.mri_y, marker = '+', c = self.colors)
            self.ax2.scatter(self.ct_x, self.ct_y, marker = '+', c = self.colors)
        
            
        if self.flag_markers == 1:
            self.markers_mri, = self.ax.plot(self.mri_x,self.mri_y, marker = 'o',ms = self.msize, mfc = 'red', linewidth = 0)
            self.markers_ct, = self.ax2.plot(self.ct_x,self.ct_y, marker = 'o',ms = self.msize, mfc = 'red', linewidth = 0)
            self.homography()
            
            self.im_mril = self.ax3.imshow(self.mri[self.ind], cmap='jet', alpha = 1)
            self.im_ctl = self.ax3.imshow(self.homography_image[self.ind], cmap='gray', alpha = self.alpha)
            
            self.im_ctr = self.ax4.imshow(self.homography_image[self.ind], cmap='jet', alpha = 1)
            self.im_mrir = self.ax4.imshow(self.mri[self.ind], cmap='gray', alpha = self.alpha)
            
        self.weighted_center_of_mass_middle_ct, self.area_middle_ct = self.brain_properties(self.noise_removal_ct(self.ct[self.z]))
        self.weighted_center_of_mass_middle_mri, self.area_middle_mri = self.brain_properties(self.mri[self.z])
            
    def on_click(self, event):
        if self.flag_markers == 1:
            if event.button == 2:  # 2 is for middle mouse button
                # get mouse cursor coordinates in pixels:
                x = event.x
                y = event.y
                # get markers xy coordinate in pixels:
                self.xydata_mri = self.ax.transData.transform(self.markers_mri.get_xydata())
                self.xdata_mri, self.ydata_mri = self.xydata_mri.T
                
                self.xydata_ct = self.ax2.transData.transform(self.markers_ct.get_xydata())
                self.xdata_ct, self.ydata_ct = self.xydata_ct.T
                # compute the linear distance between the markers and the cursor:
                self.r_mri = ((self.xdata_mri - x)**2 + (self.ydata_mri - y)**2)**0.5
                if np.min(self.r_mri) < self.msize:
                    # save figure background:
                    self.markers_mri.set_visible(False)
                    self.renderer = self.figure.canvas.renderer
                    self.figure.draw(self.renderer)
                    self.background_mri = self.figure.canvas.copy_from_bbox(self.ax.bbox)
                    self.markers_mri.set_visible(True)
                    
                    
                    self.update()
                    # store index of draggable marker:
                    self.draggable_mri = np.argmin(self.r_mri)
                else:
                    self.draggable_mri = None
                    
                self.r_ct = ((self.xdata_ct - x)**2 + (self.ydata_ct - y)**2)**0.5
                if np.min(self.r_ct) < self.msize:
                    # save figure background:
                    self.markers_ct.set_visible(False)
                    self.renderer = self.figure.canvas.renderer
                    self.figure.draw(self.renderer)
                    self.background_ct = self.figure.canvas.copy_from_bbox(self.ax2.bbox)
                    self.markers_ct.set_visible(True)
                    self.update()
                    # store index of draggable marker:
                    self.draggable_ct = np.argmin(self.r_ct)
                else:
                    self.draggable_ct = None
                
           
    def on_motion(self, event):
        if self.flag_markers == 1:
            if self.draggable_mri is not None:
                if event.xdata and event.ydata:
                    # get markers coordinate in data units:
                    self.xydata_mri = self.markers_mri.get_data() 
                    self.xdata_mri = self.xydata_mri[0] + 0
                    self.ydata_mri = self.xydata_mri[1] + 0
                    # change the coordinate of the marker that is
                    # being dragged to the ones of the mouse cursor:
                    self.xdata_mri[self.draggable_mri] = event.xdata
                    self.ydata_mri[self.draggable_mri] = event.ydata
                    # update the data of the artist:
                    self.markers_mri.set_xdata(self.xdata_mri)
                    self.markers_mri.set_ydata(self.ydata_mri)
                    # update the plot:
                    self.figure.canvas.restore_region(self.background_mri)
                    
                    self.update()
                    
            if self.draggable_ct is not None:
                if event.xdata and event.ydata:
                    # get markers coordinate in data units:
                    # self.xdata_ct, self.ydata_ct = self.markers_ct.get_data() 
                    self.xydata_ct = self.markers_ct.get_data() 
                    self.xdata_ct = self.xydata_ct[0] + 0
                    self.ydata_ct = self.xydata_ct[1] + 0
                    # change the coordinate of the marker that is
                    # being dragged to the ones of the mouse cursor:
                    self.xdata_ct[self.draggable_ct] = event.xdata
                    self.ydata_ct[self.draggable_ct] = event.ydata
                    # update the data of the artist:
                    self.markers_ct.set_xdata(self.xdata_ct)
                    self.markers_ct.set_ydata(self.ydata_ct)
                    # update the plot:
                    self.figure.canvas.restore_region(self.background_ct)
                    self.update()
                    
        
 
    def on_release(self, event):
        if self.flag_markers == 1:
            self.draggable_mri = None
            self.draggable_ct = None
            self.homography()
            self.im_ctl.set_data(self.homography_image[self.ind])
            self.im_ctr.set_data(self.homography_image[self.ind])
            self.flag_homo = True
            self.update()
    
    def on_type(self, event):
        
        if event.key == 'x':
            self.flag_propagation[self.ind] = 1
            print('Image markers locked')
            self.mri_data[:,:,self.ind] = self.markers_mri.get_xydata()
            self.ct_data[:,:,self.ind] = self.markers_ct.get_xydata()
            
        if event.key == 'c':
            self.flag_propagation[self.ind] = 0
            print('Image markers unlocked')
            self.mri_data[:,:,self.ind] = self.markers_mri.get_xydata()
            self.ct_data[:,:,self.ind] = self.markers_ct.get_xydata()
        
        if event.key == 'up' and self.ind<self.z*2:
            self.ind = (self.ind + 1) % self.slices_mri
            self.flag_arrow = True
            self.homography()
            self.set_data()
        if event.key == 'down' and self.ind>0:
            self.ind = (self.ind - 1) % self.slices_mri
            self.flag_arrow = True
            self.homography()
            self.set_data()
            
        if event.key == '+' and self.alpha<1:
            self.alpha = self.alpha + 0.1
            self.im_ctl.set_alpha(self.alpha)
            self.im_mrir.set_alpha(self.alpha)
            self.flag_homo = True
            
        if event.key == '-' and self.alpha>0:
            self.alpha = self.alpha - 0.1
            self.im_ctl.set_alpha(self.alpha)
            self.im_mrir.set_alpha(self.alpha)
            self.flag_homo = True
        
        if event.key == '8':
            T = np.float32([[1, 0, 0], [0, 1, -1]])
            self.homography_image[self.ind] = cv2.warpAffine(self.homography_image[self.ind], T, (self.homography_image.shape[1],self.homography_image.shape[2]))
            self.flag_homo = True
            self.transformations[self.ind].append('t_up')
            self.set_data()
            
        if event.key == '2':
            T = np.float32([[1, 0, 0], [0, 1, 1]])
            self.homography_image[self.ind] = cv2.warpAffine(self.homography_image[self.ind], T, (self.homography_image.shape[1],self.homography_image.shape[2]))
            self.flag_homo = True
            self.transformations[self.ind].append('t_down')
            self.set_data()
            
        if event.key == '4':
            T = np.float32([[1, 0, -1], [0, 1, 0]])
            self.homography_image[self.ind] = cv2.warpAffine(self.homography_image[self.ind], T, (self.homography_image.shape[1],self.homography_image.shape[2]))
            self.flag_homo = True
            self.transformations[self.ind].append('t_left')
            self.set_data()
            
        if event.key == '6':
            T = np.float32([[1, 0, 1], [0, 1, 0]])
            self.homography_image[self.ind] = cv2.warpAffine(self.homography_image[self.ind], T, (self.homography_image.shape[1],self.homography_image.shape[2]))
            self.flag_homo = True
            self.transformations[self.ind].append('t_right')
            self.set_data()
        
        if event.key == '9':
            self.angle = -1
            self.rotate_image()
            self.flag_homo = True
            self.transformations[self.ind].append('r_right')
            self.set_data()
        
        if event.key == '7': 
            self.angle = 1
            self.rotate_image()
            self.flag_homo = True
            self.transformations[self.ind].append('r_left')
            self.set_data()
            
        self.update()

    def on_press(self,event):
        if event.key == 'enter':
            np.save(self.path + '_selected points' +'_mri', self.mri_data)
            np.save(self.path + '_selected points' + '_ct', self.ct_data)
            np.save(self.path + '_new_ct', self.homography_image)
            np.save(self.path + '_flag_propagation.npy', self.flag_propagation)
            with open(self.path + '_transformations.data', 'wb') as filehandle:
                pickle.dump(self.transformations, filehandle)
            print('Data saved')
            
        if event.key == ' ':
            
            self.flag_markers = 1
            self.ax.cla()
            self.ax2.cla()
            self.im_mri = self.ax.imshow(self.mri[self.ind])
            self.im_ct = self.ax2.imshow(self.ct[self.ind])
            self.markers_mri, = self.ax.plot(self.mri_x,self.mri_y, marker = 'o',ms = self.msize, mfc = 'red', linewidth = 0)
            self.markers_ct, = self.ax2.plot(self.ct_x,self.ct_y, marker = 'o',ms = self.msize, mfc = 'red', linewidth = 0)
            self.mri_data[:,:,self.ind] = self.markers_mri.get_xydata()
            self.ct_data[:,:,self.ind] = self.markers_ct.get_xydata()+0
            self.homography()
            self.im_mril = self.ax3.imshow(self.mri[self.ind], cmap='jet', alpha = 1)
            self.im_ctl = self.ax3.imshow(self.homography_image[self.ind], cmap='gray', alpha = self.alpha)
            
            self.im_ctr = self.ax4.imshow(self.homography_image[self.ind], cmap='jet', alpha = 1)
            self.im_mrir = self.ax4.imshow(self.mri[self.ind], cmap='gray', alpha = self.alpha)
            self.flag_homo = True
            
            self.update()
            
    def homography(self):  
        if self.flag_propagation[self.ind]:
            self.markers_ct.set_xdata(self.ct_data[:,0,self.ind])
            self.markers_ct.set_ydata(self.ct_data[:,1,self.ind])
            
            self.markers_mri.set_xdata(self.mri_data[:,0,self.ind])
            self.markers_mri.set_ydata(self.mri_data[:,1,self.ind])
        else: 
            self.mri_data[:,:,self.ind] = self.markers_mri.get_xydata()
            self.ct_data[:,:,self.ind] = self.markers_ct.get_xydata() 
            if self.flag_arrow:
                try:
                    ct_noise_removed = self.noise_removal_ct(self.ct[self.ind])
                    self.pts_src = self.point_propagation(self.weighted_center_of_mass_middle_mri, self.area_middle_mri, self.mri_data[:,:,self.z], self.mri[self.ind])
                    self.pts_dst = self.point_propagation(self.weighted_center_of_mass_middle_ct, self.area_middle_ct, self.ct_data[:,:,self.z], ct_noise_removed)
                    self.mri_data[:,:,self.ind] = self.pts_src
                    self.ct_data[:,:,self.ind] = self.pts_dst
                except:
                    self.pts_src = self.mri_data[:,:,self.ind]
                    self.pts_dst = self.ct_data[:,:,self.ind]
            else:
                self.pts_src = self.mri_data[:,:,self.ind]
                self.pts_dst = self.ct_data[:,:,self.ind]
            # print(self.xydata_ct[5,:])
            self.h, status = cv2.findHomography(self.pts_dst, self.pts_src)
            self.homography_image[self.ind] = cv2.warpPerspective(self.ct[self.ind], self.h,(self.mri.shape[2],self.mri.shape[1]))
        
        
    def set_data(self):
        self.im_mri.set_data(self.mri[self.ind])
        self.im_ct.set_data(self.ct[self.ind])
        
        self.im_mril.set_data(self.mri[self.ind])
        self.im_ctl.set_data(self.homography_image[self.ind])
        
        self.im_mrir.set_data(self.mri[self.ind])
        self.im_ctr.set_data(self.homography_image[self.ind])
        
    def noise_removal_ct(self, image):
        self.threshold_value = filters.threshold_otsu(image)
        labeled_foreground = label((image > self.threshold_value).astype(int))
        
        if np.max(labeled_foreground)>1:
            pom = np.histogram(labeled_foreground, bins = np.arange(0,np.max(labeled_foreground)+2))
            values = pom[0][1:]    
            inds = np.argsort(-1*values)
            if self.ind!=self.z:
                ii = 0
                while ii<(len(inds)-1):
                    if (values[inds[ii]]/values[inds[ii+1]])<10:
                        ind = inds[ii]+1
                        properties_ct = regionprops((labeled_foreground==(ind))*1, image)
                        self.weighted_center_of_mass_ct = properties_ct[0].weighted_centroid
                        if abs(self.weighted_center_of_mass_ct[0]-self.weighted_center_of_mass_middle_ct[0])>100 or abs(self.weighted_center_of_mass_ct[1]-self.weighted_center_of_mass_middle_ct[1])>100:
                            ii = ii+1
                        else:
                            break
                    else:
                        ind = inds[ii]+1
                        break
            else:
                ind = inds[0]+1
        
        else: 
            ind = 1
        
        ct_removed_noise = np.array(image*(labeled_foreground==ind)*1)
        return ct_removed_noise
        
    def brain_properties(self, image):
        labeled_foreground = (image > 0).astype(int)
        properties = regionprops(labeled_foreground, image)
        weighted_center_of_mass = properties[0].weighted_centroid
        area = properties[0].area
        return weighted_center_of_mass, area

    def point_propagation(self, weighted_center_of_mass_middle, area_middle, data, image):
        
        x0 = weighted_center_of_mass_middle[1]
        y0 = weighted_center_of_mass_middle[0]
        
        weighted_center_of_mass, area = self.brain_properties(image)
    
        x01 = weighted_center_of_mass[1]
        y01 = weighted_center_of_mass[0]
        
        x1 = np.zeros(8)
        y1 = np.zeros(8)
        
        x_data = data[:,0]
        y_data = data[:,1]
                
        for ii in range(8):
            x1[ii] = (x_data[ii]-x0)/np.sqrt(area_middle/area) + x01
            y1[ii] = (y_data[ii]-y0)/np.sqrt(area_middle/area) + y01
        
        pts = np.array((x1,y1)).T
        return pts
      
    def rotate_image(self):
      image_center = tuple(np.array(self.homography_image[self.ind].shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.0)
      self.homography_image[self.ind] = cv2.warpAffine(self.homography_image[self.ind], rot_mat, self.homography_image[self.ind].shape[1::-1], flags=cv2.INTER_LINEAR)

    def mutual_information(self):
        # print('usao')
        pom = np.max([np.max(self.mri[self.ind].astype(np.uint16)),np.max(self.homography_image[self.ind].astype(np.uint16))])  
        hist = np.histogram2d(self.mri[self.ind].ravel().astype(np.uint16),self.homography_image[self.ind].ravel().astype(np.uint16), bins = pom+1, range=[[0,pom],[0,pom]])[0]
    
        #join probability mass function
        Pmc = hist / float(np.sum(hist))
         
        # marginal probability mass function
        Pm = np.sum(Pmc, axis=1) 
        Pc = np.sum(Pmc, axis=0) 
         
        # No zero values
        indNoZero = Pmc != 0
        Pmc = Pmc[indNoZero]
        indNoZero1 = Pm != 0
        Pm = Pm[indNoZero1]
        indNoZero2 = Pc != 0
        Pc = Pc[indNoZero2]
         
        # entropy
        Hmc = - np.sum(Pmc*np.log(Pmc))
        Hm = -np.sum(Pm*np.log(Pm))
        Hc = -np.sum(Pc*np.log(Pc))
         
        # mutal information
        Imc = Hm+Hc-Hmc
        return Imc
         
    def update(self):
        if self.flag_markers == 1:
         
         self.ax2.draw_artist(self.markers_ct)
         self.ax.draw_artist(self.markers_mri)
         self.mi = self.mutual_information()
         text = 'MI = '+ str(round(self.mi,2))
         self.suptitle.set_text(self.participant +'\n'+text)
         self.figure.canvas.blit(self.ax.bbox)
         self.figure.canvas.blit(self.ax2.bbox)
         
         if self.flag_homo:
             
            self.ax3.draw_artist(self.im_mril)
            self.ax3.draw_artist(self.im_ctl)
            self.ax4.draw_artist(self.im_ctr)
            self.ax4.draw_artist(self.im_mrir)
            self.mi = self.mutual_information()
            text = 'MI = '+ str(round(self.mi,2))
            self.suptitle.set_text(self.participant +'\n'+text)
            self.im_mri.axes.figure.canvas.draw()
            self.figure.canvas.blit(self.ax3.bbox)
            self.figure.canvas.blit(self.ax4.bbox)
            self.flag_homo = False
            
            
        if self.flag_arrow:
            self.ax.set_ylabel(str(self.ind))
            self.im_mri.axes.figure.canvas.draw()
            self.ax2.set_ylabel(str(self.ind))
            self.im_ct.axes.figure.canvas.draw()
            self.ax.draw_artist(self.im_mri)
            self.ax2.draw_artist(self.im_ct)
            self.flag_homo = True
            self.flag_arrow = False
            self.update()
            
    def return_data(self):
        return self.markers_mri_ct.get_xydata()


#%%

#%% The definition of paths 
category = 'zdravi' # zdravi/patoloski
list_of_participants_path = 'D:\\Teza\\Baza\\Spisak Ispitanika.xlsx'

database_path = 'D:\\Teza\\Baza\\'
plane = 'ax\\'
participant = 'NK1_25062017' 
parameters_path = 'D:\\Teza\\Baza\\Odredjivanje tacaka - Badza.xlsx' # the file containing all crossections

parameters = pd.read_excel(list_of_participants_path, sheet_name = category)
row =  np.where(parameters['Folder_name'] == participant)

image_mri=[]
image_ct=[]

crossection_mri = int(parameters['MRI '+ plane[:-1]].values[row])
crossection_ct = int(parameters['CT '+ plane[:-1]].values[row])

path_mri = database_path + category + '\\' + participant + '\\MR\\' + plane
path_ct = database_path + category + '\\' + participant + '\\CT\\' + plane

#%%

if participant == "AG1_09082018":
    file_names_mri=os.listdir(path_mri)
    crossection_mri = crossection_mri-1*np.sign(crossection_mri)
    
else: 
    file_names_mri=os.listdir(path_mri)[1:]
    crossection_mri = crossection_mri-2*np.sign(crossection_mri)

if participant == "KM1_30012020":
    file_names_ct = os.listdir(path_ct)
    crossection_ct = crossection_ct-1
    
else: 
    file_names_ct=os.listdir(path_ct)[1:]
    crossection_ct = crossection_ct-2



print('Loading MR images')
for file_mri in range(len(file_names_mri)):
    if crossection_mri<0:
        ds = pydicom.dcmread(path_mri+file_names_mri[len(file_names_mri)-file_mri-1])
    else:
        ds = pydicom.dcmread(path_mri+file_names_mri[file_mri])
    image_mri.append(ds.pixel_array)
image_mri=np.array(image_mri)

print('Loading CT images')
for file_ct in file_names_ct:
    ds = pydicom.dcmread(path_ct+file_ct)
    ct_pom = ds.pixel_array
    # image_ct.append(ct_pom)
    image_ct.append(cv2.resize(ct_pom,(image_mri.shape[1],image_mri.shape[2])))
image_ct=np.array(image_ct)

print('Change of crossection')
n = image_mri.shape[0]
mri = change_middle_plane(image_mri, crossection_mri, n)
ct = change_middle_plane(image_ct, crossection_ct, n)
n = mri.shape[0]
z = round(n/2)

path_to_save = database_path + 'Saved data\\' + category + '\\' +participant+'\\' + participant +'_'+ plane[:-1]
#%%

fig = plt.figure(figsize=(10,10))

myfigure = MyFigureCanvas(fig,participant,ct,mri,z, path_to_save)
fig.canvas.mpl_connect('motion_notify_event', myfigure.on_motion)
fig.canvas.mpl_connect('button_press_event', myfigure.on_click)
fig.canvas.mpl_connect('button_release_event', myfigure.on_release)
fig.canvas.mpl_connect('key_press_event', myfigure.on_press)
fig.canvas.mpl_connect('key_press_event', myfigure.on_type)

# my_data = myfigure.return_data()


    
