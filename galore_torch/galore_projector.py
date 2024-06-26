import torch
import bitsandbytes.functional as bnbf

# We document only the changes we made (e.g. adding extra arguments) to the GaLore pre-release

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std', quantize_type='8bit_blockwise', quantize_blocksz=4096):
        '''
        We added the arguments:
        quantize_type: '4bit', '8bit', '8bit_blockwise' or None; method for quantizing projection matrix
        quantize_blocksz: Block size for projection matrix quantization
        '''
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type
        self.quantize_type = quantize_type if quantize_type in ['4bit', '8bit', '8bit_blockwise'] else None
        self.quantize_blocksz = quantize_blocksz

    def project(self, full_rank_grad, iter):
        '''
        Before matrix-multiplying the projection matrix (self.ortho_matrix) with the gradient matrix (self.full_rank_grad),
        we dequantize self.ortho_matrix. After performing the matrix multiplication, we quantize 
        self.ortho_matrix back for storage and save its state.
        '''
        if self.proj_type == 'std': 
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                    if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)  
                if self.quantize_type: self.ortho_matrix = self._dequantize(self.ortho_matrix, self.quantize_type, self.quantization_state, full_rank_grad.dtype) # dequantize
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
                if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                    if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)  
                if self.quantize_type: self.ortho_matrix = self._dequantize(self.ortho_matrix, self.quantize_type, self.quantization_state, full_rank_grad.dtype) # dequantize         
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
                if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)
                
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                    if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)
                if self.quantize_type: self.ortho_matrix = self._dequantize(self.ortho_matrix, self.quantize_type, self.quantization_state, full_rank_grad.dtype)
                low_rank_grad = torch.matmul(self.ortho_matrix.t(),full_rank_grad)
                if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                    if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)
                if self.quantize_type: self.ortho_matrix = self._dequantize(self.ortho_matrix, self.quantize_type, self.quantization_state, full_rank_grad.dtype)
                low_rank_grad = torch.matmul(full_rank_grad,self.ortho_matrix.t())
                if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)

        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)
            if self.quantize_type: self.ortho_matrix = self._dequantize(self.ortho_matrix, self.quantize_type, self.quantization_state, full_rank_grad.dtype)
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)

        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)
            if self.quantize_type: self.ortho_matrix = self._dequantize(self.ortho_matrix, self.quantize_type, self.quantization_state, full_rank_grad.dtype)
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)

        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
                if self.quantize_type:
                    self.ortho_matrix[0], self.quantization_state = self._quantize(self.ortho_matrix[0], self.quantize_type, self.quantize_blocksz)
                    self.ortho_matrix[1], self.quantization_state_full = self._quantize(self.ortho_matrix[1], self.quantize_type, self.quantize_blocksz)
            if self.quantize_type:
                    self.ortho_matrix[0] = self._dequantize(self.ortho_matrix[0], self.quantize_type, self.quantization_state, full_rank_grad.dtype)
                    self.ortho_matrix[1] = self._dequantize(self.ortho_matrix[1], self.quantize_type, self.quantization_state2, full_rank_grad.dtype)
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()
            if self.quantize_type:
                    self.ortho_matrix[0], self.quantization_state = self._quantize(self.ortho_matrix[0], self.quantize_type, self.quantize_blocksz)
                    self.ortho_matrix[1], self.quantization_state_full = self._quantize(self.ortho_matrix[1], self.quantize_type, self.quantize_blocksz)
        
        return low_rank_grad

    def project_back(self, low_rank_grad):
        '''
        Again, we dequantize self.ortho_matrix before using it in a matrix multiplication.
        Afterwards, we re-quantize it back for storage.
        '''

        if self.proj_type == 'std':
            if self.quantize_type: self.ortho_matrix = self._dequantize(self.ortho_matrix, self.quantize_type, self.quantization_state, low_rank_grad.dtype) # dequantize
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz) # quantize back for storage
        elif self.proj_type == 'reverse_std':
            if self.quantize_type: self.ortho_matrix = self._dequantize(self.ortho_matrix, self.quantize_type, self.quantization_state, low_rank_grad.dtype)
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)
        elif self.proj_type == 'right':
            if self.quantize_type: self.ortho_matrix = self._dequantize(self.ortho_matrix, self.quantize_type, self.quantization_state, full_rank_grad.dtype)
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)
        elif self.proj_type == 'left':
            if self.quantize_type: self.ortho_matrix = self._dequantize(self.ortho_matrix, self.quantize_type, self.quantization_state, full_rank_grad.dtype)
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            if self.quantize_type: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, self.quantize_type, self.quantize_blocksz)
        elif self.proj_type == 'full':
            if self.quantize_type:
                self.ortho_matrix[0] = self._dequantize(self.ortho_matrix[0], self.quantize_type, self.quantization_state, full_rank_grad.dtype)
                self.ortho_matrix[1] = self._dequantize(self.ortho_matrix[0], self.quantize_type, self.quantization_state_full, full_rank_grad.dtype)
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
            if self.quantize_type:
                self.ortho_matrix[0], self.quantization_state = self._quantize(self.ortho_matrix[0], self.quantize_type, self.quantize_blocksz)
                self.ortho_matrix[0], self.quantization_state[1] = self._quantize(self.ortho_matrix[1], self.quantize_type, self.quantize_blocksz)
        
        return full_rank_grad * self.scale
        
        
    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
            
        U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
        
        #make the smaller matrix always to be orthogonal matrix
        if type=='right':
            A = U[:, :rank] @ torch.diag(s[:rank])
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type) 
            return B
        elif type=='left':
            A = U[:, :rank]
            B = torch.diag(s[:rank]) @ Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')

    def _quantize(self, A, quantize_type, bsz=4096):
        '''
        Quantize the tensor A according to the desired method.
        '''
        if quantize_type == '8bit':
            return bnbf.quantize(A)
        elif quantize_type == '4bit':
            return bnbf.quantize_4bit(A)
        elif quantize_type == '8bit_blockwise':
            return bnbf.quantize_blockwise(A, blocksize=bsz)
        else:
            raise ValueError("Invalid quantization type")

    def _dequantize(self, A, quantize_type, state, dtype):
        '''
        Dequantize the tensor A according to the desired method.
        '''
        if quantize_type == '8bit':
            return bnbf.dequantize(A, state).to(dtype) # dequantizes to fp32 by default without .to(dtype)
        elif quantize_type == '4bit':
            return bnbf.dequantize_4bit(A, state).to(dtype)
        elif quantize_type == '8bit_blockwise':
            return bnbf.dequantize_blockwise(A, state).to(dtype)
        else:
            raise ValueError("Invalid quantization type")