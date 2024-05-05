import torch
import bitsandbytes.functional as bnbf

# Ideally, we make these command line arguments instead. However, since we are using LLama-Factory to benchmark,
# we'd have to also modify the Llama-Factory package to accomodate for the new argument. For now, we can just hardcode these flags
# and modify it in between runs, so we don't have to touch Llama-Factory.  

QUANTIZE = 'fp4' # if quantizing, choose one of 'fp8', 'fp4', 'fp8_blockwise' - otherwise, set to None
BLOCKSZ = 128 # only matters if QUANTIZE = 'fp8_blockwise'

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0,
                 proj_type='std', quantize_type='fp8', blocksize=128):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    # only added quantization functionality for 'std'
    def project(self, full_rank_grad, iter):
        if self.proj_type == 'std': 
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                    if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)  
                if QUANTIZE: self.ortho_matrix = self._dequantize(self.ortho_matrix, QUANTIZE, self.quantization_state, full_rank_grad.dtype) # dequantize
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
                if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                    if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)  
                if QUANTIZE: self.ortho_matrix = self._dequantize(self.ortho_matrix, QUANTIZE, self.quantization_state, full_rank_grad.dtype) # dequantize         
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
                if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)
                
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                    if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)
                if QUANTIZE: self.ortho_matrix = self._dequantize(self.ortho_matrix, QUANTIZE, self.quantization_state, full_rank_grad.dtype)
                low_rank_grad = torch.matmul(self.ortho_matrix.t(),full_rank_grad)
                if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                    if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)
                if QUANTIZE: self.ortho_matrix = self._dequantize(self.ortho_matrix, QUANTIZE, self.quantization_state, full_rank_grad.dtype)
                low_rank_grad = torch.matmul(full_rank_grad,self.ortho_matrix.t())
                if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)

        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)
            if QUANTIZE: self.ortho_matrix = self._dequantize(self.ortho_matrix, QUANTIZE, self.quantization_state, full_rank_grad.dtype)
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)

        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)
            if QUANTIZE: self.ortho_matrix = self._dequantize(self.ortho_matrix, QUANTIZE, self.quantization_state, full_rank_grad.dtype)
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)

        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
                if QUANTIZE:
                    self.ortho_matrix[0], self.quantization_state = self._quantize(self.ortho_matrix[0], QUANTIZE, BLOCKSZ)
                    self.ortho_matrix[1], self.quantization_state_full = self._quantize(self.ortho_matrix[1], QUANTIZE, BLOCKSZ)
            if QUANTIZE:
                    self.ortho_matrix[0] = self._dequantize(self.ortho_matrix[0], QUANTIZE, self.quantization_state, full_rank_grad.dtype)
                    self.ortho_matrix[1] = self._dequantize(self.ortho_matrix[1], QUANTIZE, self.quantization_state2, full_rank_grad.dtype)
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()
            if QUANTIZE:
                    self.ortho_matrix[0], self.quantization_state = self._quantize(self.ortho_matrix[0], QUANTIZE, BLOCKSZ)
                    self.ortho_matrix[1], self.quantization_state_full = self._quantize(self.ortho_matrix[1], QUANTIZE, BLOCKSZ)
        
        return low_rank_grad

    def project_back(self, low_rank_grad):

        if self.proj_type == 'std':
            if QUANTIZE: self.ortho_matrix = self._dequantize(self.ortho_matrix, QUANTIZE, self.quantization_state, low_rank_grad.dtype) # dequantize
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ) # quantize back for storage
        elif self.proj_type == 'reverse_std':
            if QUANTIZE: self.ortho_matrix = self._dequantize(self.ortho_matrix, QUANTIZE, self.quantization_state, low_rank_grad.dtype)
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)
        elif self.proj_type == 'right':
            if QUANTIZE: self.ortho_matrix = self._dequantize(self.ortho_matrix, QUANTIZE, self.quantization_state, full_rank_grad.dtype)
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)
        elif self.proj_type == 'left':
            if QUANTIZE: self.ortho_matrix = self._dequantize(self.ortho_matrix, QUANTIZE, self.quantization_state, full_rank_grad.dtype)
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            if QUANTIZE: self.ortho_matrix, self.quantization_state = self._quantize(self.ortho_matrix, QUANTIZE, BLOCKSZ)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
        

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
        # print("QUANTIZING")
        if quantize_type == 'fp8':
            return bnbf.quantize(A)
        elif quantize_type == 'fp4':
            return bnbf.quantize_4bit(A)
        elif quantize_type == 'fp8_blockwise':
            return bnbf.quantize_blockwise(A, blocksize=bsz)
        else:
            raise ValueError("Invalid quantization type")

    def _dequantize(self, A, quantize_type, state, dtype):
        # print("DEQUANTIZING")
        if quantize_type == 'fp8':
            return bnbf.dequantize(A, state).to(dtype) # dequantizes to fp32 by default without .to(dtype)
        elif quantize_type == 'fp4':
            return bnbf.dequantize_4bit(A, state).to(dtype)
        elif quantize_type == 'fp8_blockwise':
            return bnbf.dequantize_blockwise(A, state).to(dtype)
        else:
            raise ValueError("Invalid quantization type")