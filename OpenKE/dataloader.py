#coding:utf-8
import numpy as np
import os
import ctypes

def allocate_array(shape, dtype):
    # Allocate a numpy array for passing data between Python and C
    array = np.zeros(shape, dtype=dtype)
    addr = array.__array_interface__['data'][0]
    return array,addr

class DataLoader:
    '''
    Call out to the C functions to load and sample data.
    '''

    def __init__(self,
            data_path:str,
            lib_path:str=None,
            n_batches:int=100,
            negative_ent:int=1,
            negative_rel:int=0,
            bern:int=0,
            work_threads:int=8):

        self.negative_ent = negative_ent
        self.negative_rel = negative_rel
        self.n_batches = n_batches

        if lib_path is None:
            # XXX This is a bit hacky
            # Find the object file
            import OpenKE
            dp = os.path.dirname(OpenKE.__file__)
            for fn in os.listdir(dp):
                if not fn.startswith("libdataloader"): continue
                if not fn.endswith(".so"): continue
                break
            else:
                raise Exception("Can't find the data loader object file")
            lib_path = os.path.join(dp,fn)

        # The C library will segfault if the path doesn't end in a slash
        if not data_path.endswith('/'):
            data_path += '/'
    
        # Load the library and configure it
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.lib.setInPath(ctypes.create_string_buffer(data_path.encode(), len(data_path) * 2))
        self.lib.setBern(bern)
        self.lib.setWorkThreads(work_threads)
        self.lib.randReset()
        self.lib.importTrainFiles()

        '''
        C library exposes the following functions
        '''

        # void sampling(INT *batch_h, INT *batch_t, INT *batch_r,
        #               REAL *batch_y, INT batchSize, INT negRate = 1,
        #               INT negRelRate = 0)
        self.lib.sampling.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64]

        # void getHeadBatch(INT *ph, INT *pt, INT *pr)
        self.lib.getHeadBatch.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        # void getTailBatch(INT *ph, INT *pt, INT *pr)
        self.lib.getTailBatch.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        
        # void testHead(REAL *con)
        self.lib.testHead.argtypes = [ctypes.c_void_p]

        # void testTail(REAL *con)
        self.lib.testTail.argtypes = [ctypes.c_void_p]

        # void getTestBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt,
        #                   INT *nr)
        self.lib.getTestBatch.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        # void getValidBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt,
        #                    INT *nr)
        self.lib.getValidBatch.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        # void getBestThreshold(REAL *relThresh, REAL *score_pos,
        #                       REAL *score_neg)
        self.lib.getBestThreshold.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        # void test_triple_classification(REAL *relThresh, REAL *score_pos,
        #                                 REAL *score_neg)
        self.lib.test_triple_classification.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    @property
    def batch_size(self):
        return int(self.n_train_samples / self.n_batches)

    @property
    def n_negative(self):
        return self.negative_ent + self.negative_rel

    @property
    def n_entities(self):
        return self.lib.getEntityTotal()

    @property
    def n_relations(self):
        return self.lib.getRelationTotal()

    @property
    def n_train_samples(self):
        return self.lib.getTrainTotal()

    @property
    def n_test_samples(self):
        return self.lib.getTestTotal()

    @property
    def n_validation_samples(self):
        return self.lib.getValidTotal()

    @property
    def batch_seq_size(self):
        return self.batch_size * (1 + self.n_negative)

    def sample(self):
        if not hasattr(self, 'sample_arrays'):
            shape = self.batch_seq_size
            self.sample_arrays = {
                'h': allocate_array(shape, np.int64),
                't': allocate_array(shape, np.int64),
                'r': allocate_array(shape, np.int64),
                'y': allocate_array(shape, np.float32),
            }

        # Call the sampling function.  This updates the value of the batch_X
        # arrays.
        self.lib.sampling(
                self.sample_arrays['h'][1],
                self.sample_arrays['t'][1],
                self.sample_arrays['r'][1],
                self.sample_arrays['y'][1],
                self.batch_size,
                self.negative_ent,
                self.negative_rel)

        # Return updated arrays
        # NOTE:  This is a pass by reference, so copy if you want to keep these
        #        data
        return (self.sample_arrays['h'][0],
                self.sample_arrays['t'][0],
                self.sample_arrays['r'][0],
                self.sample_arrays['y'][0])
