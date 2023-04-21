#include <iostream>
#include <ctime>
#include <chrono>
#include <vector>

const int THREADS_PER_BLOCK = 1024;
const int size = 1000;

void randomly_initialize_model(int* model, int size){
    for(int y = 0; y < size; y++)
        for(int x = 0; x < size; x++)
            model[y*size + x] = (rand() > RAND_MAX/2) ? 1 : -1;
}

void display_model(int* model, int size){
    for(int y = 0; y < size; y++){
        for(int x = 0; x < size; x++){
            std::cout.width(3);
            std::cout << model[y*size + x] << ' ';
        }
        std::cout << std::endl;
    }
}

int* allocate_model(int size){
    int* model = new int[size*size]; 
    return model;
}

void deallocate_model(int* model){
    delete[] model;
}

void swap_models(int*& model_1, int*& model_2){
    int* temp_model;
    temp_model = model_1;
    model_1 = model_2;
    model_2 = temp_model;
}

int* allocate_model_d(int size){
    int* model_d;
    cudaMalloc((void**)&model_d, size * size * sizeof(int));
    return model_d;
}

void deallocate_model_d(int* model){
    cudaFree(model);
}

void copy_model_to_device(int* model, int* model_d, int size){
    cudaMemcpy(model_d, model, size * size * sizeof(int), cudaMemcpyHostToDevice);
}

void copy_model_to_cpu(int* model, int* model_d, int size){
    cudaMemcpy(model, model_d, size * size * sizeof(int), cudaMemcpyDeviceToHost);
}

void ising_v0(int* model, int size, int iterations){
    int* auxiliary_model = allocate_model(size);
    for(int i = 0; i < iterations; i++){
        for(int y = 0; y < size; y++)
            for(int x = 0; x < size; x++)
                auxiliary_model[y*size + x] = ((model[y*size + x] + model[x != 0 ? y*size + x - 1 : y*size + x - 1 + size] + model[x != size - 1 ? y*size + x + 1 : y*size + x + 1 - size] + model[y != 0 ? (y-1)*size + x : (size-1)*size + x] + model[y != size - 1 ? (y+1)*size + x : x]) > 0) ? 1 : -1;
        swap_models(model, auxiliary_model);
    }
    deallocate_model(auxiliary_model);
}

void test_ising_performance_v0(int* ising_model, int size, int iterations){
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    ising_v0(ising_model, size, iterations);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << std::endl;
    //display_model(ising_model, size);
}

__global__ void ising_iteration_v1(int* model, int* auxiliary_model, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size*size)
        auxiliary_model[index] = (model[index] + model[index % size != 0 ? index - 1 : index -1 + size] + model[index % (size - 1) != 0 ? index + 1 : index + 1 - size] + model[index >= size ? index - size : (size - 1)*size + index] + model[index < size*(size - 1) ? index + size : index - size*(size - 1)]) > 0 ? 1 : -1;
}

void test_ising_performance_v1(int* ising_model, int size, int iterations){
    std::chrono::high_resolution_clock::time_point start, end;
    int* model_d = allocate_model_d(size);
    int* auxiliary_model_d = allocate_model_d(size);
    copy_model_to_device(ising_model, model_d, size);
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++){
        ising_iteration_v1<<<(size*size)/THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(model_d, auxiliary_model_d, size);
        swap_models(model_d, auxiliary_model_d);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << std::endl;
    copy_model_to_cpu(ising_model, model_d, size);
    deallocate_model_d(model_d);
    deallocate_model_d(auxiliary_model_d);
}

__global__ void ising_iteration_v2(int* model, int* auxiliary_model, int size, int elements_per_thread){
    int starting_index = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
    int index;
    for(int offset = 0; offset < elements_per_thread; offset++){
        index = starting_index + offset;
        if(index < size*size)
            auxiliary_model[index] = (model[index] + model[index % size != 0 ? index - 1 : index -1 + size] + model[index % (size - 1) != 0 ? index + 1 : index + 1 - size] + model[index >= size ? index - size : (size - 1)*size + index] + model[index < size*(size - 1) ? index + size : index - size*(size - 1)]) > 0 ? 1 : -1;
    }
}

void ising_v2(int* model, int size, int iterations, int elements_per_thread){ 
    std::chrono::high_resolution_clock::time_point start, end;
    int* model_d = allocate_model_d(size);
    int* auxiliary_model_d = allocate_model_d(size);
    copy_model_to_device(model, model_d, size);
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++){
        ising_iteration_v2<<<(size*size)/(THREADS_PER_BLOCK*elements_per_thread) + 1, THREADS_PER_BLOCK>>>(model_d, auxiliary_model_d, size, elements_per_thread);
        swap_models(model_d, auxiliary_model_d);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << std::endl;
    copy_model_to_cpu(model, model_d, size);
    deallocate_model_d(model_d);
    deallocate_model_d(auxiliary_model_d);
}

void test_ising_performance_v2(int* ising_model, int size, int iterations, int elements_per_thread){
    ising_v2(ising_model, size, iterations, elements_per_thread);
    //display_model(ising_model, size);    
}

__global__ void ising_iteration_v3(int* model, int* auxiliary_model){
    __shared__ int temp[THREADS_PER_BLOCK + 2*size];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size*size){
        temp[size + threadIdx.x] = model[index];
        if(threadIdx.x < size){
            if(index < size)
                temp[threadIdx.x] = model[size*(size - 1) + index];
            else
                temp[threadIdx.x] = model[index - size];
        }
        if(threadIdx.x >= THREADS_PER_BLOCK - size){
            if(index >= size*(size-1))
                temp[threadIdx.x + 2*size] = model[index - (size*(size-1))];
            else
                temp[threadIdx.x + 2*size] = model[index + size];
        }
    }
    __synchthreads();
    if(index < size*size){
        auxiliary_model[index] = (temp[size + threadIdx.x] + temp[threadIdx.x] + temp[threadIdx.x + 2*size] +
                                 temp[index % size != 0 ? size - 1 + threadIdx.x : 2*size - 1 + threadIdx.x] +
                                 temp[index % (size - 1) != 0 ? size + 1 + threadIdx.x : threadIdx.x + 1]) > 0 ? 1 : -1;
    }
}

void ising_v3(int* model, int iterations){ 
    std::chrono::high_resolution_clock::time_point start, end;
    int* model_d = allocate_model_d(size);
    int* auxiliary_model_d = allocate_model_d(size);
    copy_model_to_device(model, model_d, size);
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++){
        ising_iteration_v3<<<(size*size)/THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(model_d, auxiliary_model_d);
        swap_models(model_d, auxiliary_model_d);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << std::endl;
    copy_model_to_cpu(model, model_d, size);
    deallocate_model_d(model_d);
    deallocate_model_d(auxiliary_model_d);
}

void test_ising_performance_v3(int* ising_model, int iterations){
    ising_v3(ising_model, iterations);
    //display_model(ising_model, size);    
}

void copy_model(int* source_model, int* destination_model, int size){
    for(int i = 0; i < size*size; i++)
        destination_model[i] = source_model[i];
}

bool are_models_equal(int* model_1, int* model_2, int size){
    for(int i = 0; i < size*size; i++)
        if(model_1[i] != model_2[i])
            return false;
    return true;
}

int main(int argc, char** argv){
    srand(time_t(time(NULL)));
    //if(argc != 4) exit(1);
    //int size, iterations, elements_per_thread;
    //size = atoi(argv[1]);
    //iterations = atoi(argv[2]);   
    //elements_per_thread = atoi(argv[3]);
    
    std::vector<int> sizes{100, 500, 1000, 10000};
    std::vector<int> iterations{1, 10, 100, 1000, 10000};
    std::vector<int> elements_per_thread_vector{1, 10, 100, 1000};

    int* ising_model_1 = allocate_model(size);
    int* ising_model_2 = allocate_model(size);
    for(int iteration: iterations){
        randomly_initialize_model(ising_model_1, size);
        copy_model(ising_model_1, ising_model_2, size);
        test_ising_performance_v3(ising_model_1, iterations);
        test_ising_performance_v1(ising_model_1, size, iterations);
    }
    deallocate_model(ising_model_1);
    deallocate_model(ising_model_2);
}
