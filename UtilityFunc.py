# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:11:51 2018

@author: Riza
"""

#%% Modules:

import os
import shutil
import numbers
import warnings

import numpy as np
import numpy.linalg as la
shape = np.shape
size = np.size
reshape = np.reshape

import scipy.sparse as sparse

#%% Utility Function Class:

class UF():
    """Class to define custom functions to facilitate using Python."""

    def isnumber(self, x):
        """Checks if x contains all numbers."""
        if type(x)==list:
            x = self.unpackList(x)
            for i in range(len(x)):
                if not isinstance(x[i], numbers.Number):
                    return False
            return True
        
        elif type(x)==np.ndarray:
            x = reshape(x, size(x))
            for i in range(size(x)):
                if not isinstance(x[i], numbers.Number):
                    return False
            return True
        
        else:
            return isinstance(x, numbers.Number)
    
    
    def isempty(self, x):
        """Checks if x is empty."""
        if type(x)==list and len(x)>0:
            return False
        elif type(x)==dict and len(x)>0:
            return False
        elif not (type(x)==list or type(x)==dict) and size(x)>0:
            return False
        else:
            return True
        
        
    def isnone(self, x):
        """Checks if x is none. Returns true even if a single element is None."""
        
        if self.isempty(x):
            return False
        
        elif type(x)==list:
            x = self.unpackList(x)
            for i in range(len(x)):
                if type(x[i]).__module__==np.__name__ and size(x[i])>1 and (x[i]==None).any():
                    return True
                elif type(x[i]).__module__==np.__name__ and size(x[i])==1 and x[i]==None:
                    return True
                elif type(x[i]).__module__==np.__name__:
                    continue
                elif x[i]==None:
                    return True
            return False
        
        elif type(x).__module__==np.__name__ and size(x)>1:
            return (x==None).any()
        
        else:
            return x==None
        
        
    def unpackList(self, x):
        """
        Retruns elements of a list with arbitraty structure, i.e., a list of 
        lists, in a single list.
        """
        # Handle error:
        if type(x)!=list:
            raise ValueError('Input argument must be a list!')
            
        outList = []
        for i in range(len(x)):
            if type(x[i])==list:
                outList.extend(self.unpackList(x[i]))
            elif not self.isempty(x[i]):
                outList.append(x[i])
        
        return outList
    
    
    def vstack(self, tup):
        """
        Function to stack tensors vertically. The numpy version does not accept
        any empty lists (arrays).
        """
        if self.isempty(self.unpackList(tup)): return []
        
        tup2 = []
        for i in range(len(tup)):
            if not self.isempty(tup[i]):
                tup2.append(tup[i])
                
        return np.vstack(tup2)
    
    
    def hstack(self, tup):
        """
        Function to stack tensors horizontally. The numpy version does not accept
        any empty lists (arrays).
        """
        if self.isempty(self.unpackList(tup)): return []
        
        tup2 = []
        for i in range(len(tup)):
            if not self.isempty(tup[i]):
                tup2.append(tup[i])
        return np.hstack(tup2)


    def nodeNum(self, x, val):
        """
        Function to find closest value in the vector 'x' to value 'val'.
        
        Inputs:
            x [nx1]: vector of values
            val [mx1]: vector of values to be found in vector 'x'
        """
        # Handle errors:
        if not type(x)==np.ndarray:
            raise ValueError('x must be a vector!')
        if not self.isnumber(val):
            raise ValueError('val must be a vector!')
        elif isinstance(val, numbers.Number):
            val = [val]
        
        ind = []
        for i in range(size(val)):
            ind.append(np.argmin(np.abs(x-val[i])))
        
        return ind
        
        
    def pairMats(self, mat1, mat2):
        """
        Utility function to pair matrices 'mat1' and 'mat2' by tiling 'mat2' and
        repeating rows of 'mat1' for each tile of 'mat1'.
        
        Inputs:
            mat1 [n1xm1]
            mat2 [n2xm2]
            
        Output:
            MAT [(n1*n2)x(m1+m2)]
        """
        # Error handling:
        if self.isempty(mat1):
            return mat2
        elif self.isempty(mat2):
            return mat1
        
        # Matrix dimensions:
        sh1 = shape(mat1)
        sh2 = shape(mat2)
        
        # Repeat one row of the first matrix per tile of second matrix:
        ind = np.arange(0, sh1[0])[np.newaxis].T
        ind = np.tile(ind, reps=[1, sh2[0]])
        ind = reshape(ind, newshape=sh1[0]*sh2[0])
        MAT1 = mat1[ind]
        
        # Tile second matrix:
        MAT2 = np.tile(mat2, reps=[sh1[0],1])
        
        return np.hstack([MAT1, MAT2])
            

    def rejectionSampling(self, func, smpfun, dof, dofT=None):
        """
        Function to implement the rejection sampling algorithm to select 
        points according to a given loss function. If more than one loss function
        is used in 'func', dofT determines the number of nodes that belong to 
        each one of them.
        
        Inputs:
            func: function handle to determine the loss value at candidate points
            smpfun: function to draw samples from
            dof [mx1]: number of samples to be drawn for each segment
            dofT [mx1]: determines the segment length in the samples and function values
        """
        # Error handling:
        if isinstance(dof, numbers.Number) and not self.isnone(dofT):
            raise ValueError('\'dofT\' must be None for scalar \'dof\'')
        elif isinstance(dof, numbers.Number):
            dof = [dof]
        m = len(dof)
            
        if m>1 and self.isnone(dofT):
            raise ValueError('\'dofT\' must be provided when \'dof\' is a list!')
        
        # Rejection sampling procedure:
        maxfunc = lambda x,i: np.max(x)
        fmax = self.listSegment(func(), dofT, maxfunc)                  # maximum function values over the uniform grid
        def rejecSmp(val, i):
            """Function for rejection sampling to be assigned to listSegment()."""
            nt = len(val)                                               # number of samples
            
            # Uniform drawing for each sample to determine its rejection or acceptance:
            uniformVal = np.random.uniform(size=[nt,1])
            
            # Rejection sampling:
            ind = uniformVal < (val/fmax[i])                            # acceptance criterion
            return reshape(ind, nt)
            
        # Initialization:
        ns = [0 for i in range(m)]                                      # number of samples
        inpuT = [[] for i in range(m)]                                  # keep accepted samples
        flag = True
        while flag:
            # draw new samples:
            samples = smpfun()
            smpList = self.listSegment(samples, dofT)
            
            # Function value at randomly sampled points:
            val = func(samples)
            
            # Rejection sampling for each segment:
            ind = self.listSegment(val, dofT, rejecSmp)                 # accepted indices
            
            flag = False                                                # stopping criterion
            for i in range(m):
                inpuTmp = smpList[i][ind[i]]                            # keep accepted samples
                inpuT[i] = self.vstack([inpuT[i], inpuTmp])             # add to previously accepted samples
                ns[i] += np.sum(ind[i])                                 # update the number of optimal samples
                if not flag and ns[i]<dof[i]: flag=True
            
        for i in range(m):
            inpuT[i] = inpuT[i][:dof[i],:]                              # keep only 'dof' samples
        
        return np.vstack(inpuT)                                         # stack all samples together
        
        
    def listSegment(self, vec, segdof, func=None):
        """
        This function segemnts a vector of values into smaller pieces stored
        in a list and possibly apply 'func' to each segment separately.
        
        Inputs:
            vec [n x dim]: vector to be segmented
            segdof [mx1]: segmentation nodes (each entry specifies the NUMBER 
                   of nodes in one segment)
            func: function to be applied to segments separately - this function
                should accept a list and its index in the original list
        """
        n = len(vec)
        
        # Error handling:
        if self.isnone(segdof) and self.isnone(func):
            return [vec]
        elif self.isnone(segdof):
            return [func(vec,0)]
        elif isinstance(segdof, numbers.Number):
            segdof = [segdof]
            m = 1
        else:
            m = len(segdof)
            
        if segdof[-1]>n:
            raise ValueError('\'segdof\' is out of bound!')
            
        # Segmentation:
        outVec = [[] for i in range(m)]
        ind = 0
        for i in range(m):
            if not self.isnone(func):
                outVec[i] = func(vec[ind:(ind+segdof[i])][:], i)
            else:
                outVec[i] = vec[ind:(ind+segdof[i])][:]
            ind += segdof[i]
            
        # Add the remainder if it exists:
        if ind<n and not self.isnone(func):
            outVec.append( func(vec[ind:], i) )
        elif ind<n:
            outVec.append(vec[ind:])
        
        return outVec
        
        
    def reorderList(self, x, ind):
        """Reorder the entries of the list 'x' according to indices 'ind'."""
        n = len(x)
        
        # Error handling:
        if not len(ind)==n:
            warnings.warn('length of the indices is not equal to the length of the list!')
        if not self.isnumber(ind):
            raise ValueError('\'ind\' must be a list of integers!')
            
        if type(ind)==np.ndarray:
            ind = reshape(ind, n)
            
        return [x[i] for i in ind]
            
        
    def buildDict(self, keys, values):
        """Build a dict with 'keys' and 'values'."""
        n = len(keys)
        
        # Error handling:
        if not len(values)==n:
            raise ValueError('length of the keys and values must match!')
            
        mydict = {}
        for i in range(n):
            mydict[keys[i]] = values[i]
            
        return mydict
        
    
    def l2Err(self, xTrue, xApp):
        """Function to compute the normalized l2 error."""
        
        # Preprocessing:
        if sparse.issparse(xTrue): xTrue = xTrue.todense()
        if sparse.issparse(xApp): xApp = xApp.todense()
        n = size(xTrue)
        if size(shape(xTrue))==1:
            xTrue = reshape(xTrue, [n,1])
        if not size(xApp)==n:
            raise ValueError('\'xTrue\' and \'xApp\' must have the same shape!')
        elif size(shape(xApp))==1:
            xApp = reshape(xApp, [n,1])
        
        return la.norm(xTrue-xApp)/la.norm(xTrue)
        
        
    def clearFolder(self, folderpath):
        """Function to remove the content of the folder specified by 'folderpath'."""
        
        if self.isempty(os.listdir(folderpath)): return
        
        # Make sure that the call to this function was intended:
        while True:
            answer = input('clear the content of the folder? (y/n)\n')
            if answer.lower()=='y' or answer.lower()=='yes':
                break
            elif answer.lower()=='n' or answer.lower()=='no':
                return
        
        for file in os.listdir(folderpath):
            path = os.path.join(folderpath, file)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(e)
        
        
    def copyFile(self, filename, folderpath):
        """
        Function to backup the operator settings for later reference.
        Inputs:
            filename: name of the current operator file
            folderpath: the destination folder path
        """
        if not os.path.exists(filename):
            filename = os.path.join(os.getcwd(), filename)
            if not os.path.exists(filename):
                raise ValueError('The file does not exist!')
        
        shutil.copy2(filename, folderpath)      # copy the file


    def polyArea(self, x, y=None):
        """
        Function to compute the area of a polygon using Shoelace formula.
        
        Inputs:
            x: vector of first coordinates or all coordinates in columns
            y: vector of second coordinates
        """
        if self.isnone(y) and not shape(x)[1]==2:
            raise ValueError('input must be 2d!')
        elif self.isnone(y):
            y = x[:,0]
            x = x[:,1]
        elif len(shape(x))>1 and not shape(x)[1]==1:
            raise ValueError('\'x\' must be a 1d vector of first coordinates!')
        elif not len(x)==len(y):
            raise ValueError('\'x\' and \'y\' must be the same length!')
        else:
            x = reshape(x, len(x))
            y = reshape(y, len(x))
        
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        
        

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        