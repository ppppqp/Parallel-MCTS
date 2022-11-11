#include <iostream>
#include <thread>
#include <vector>
#include <ctime>
using namespace std;
// double send_n (unsigned int messages_count, size_t message_size)
// {

//   std::unique_ptr<unsigned char[]> host_memory (new unsigned char[messages_count * message_size]);

//   unsigned char *device_memory {};
//   cudaMalloc (&device_memory, messages_count * message_size);
//   cudaDeviceSynchronize ();

//   auto begin = std::chrono::high_resolution_clock::now ();

//   for (unsigned int message = 0; message < messages_count; message++)
//     cudaMemcpy (
//           device_memory + message_size * message,
//           host_memory.ge () + message_size * message,
//           message_size, cudaMemcpyDeviceToHost);

//   auto end = std::chrono::high_resolution_clock::now ();

//   cudaFree (device_memory);
//   return std::chrono::duration<double> (end - begin).count ();
// }

__global__
void initialize(char *begin){
    begin[0] = 'H';
    begin[1] = 'e';
    begin[2] = 'l';
    begin[3] = 'l';
    begin[4] = 'o';
}

int main(){
    vector<thread> threads;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cout << "device count:" << device_count<< endl;
    for(unsigned int did = 0; did < device_count; did++){
        threads.push_back(thread([&, did](){
            char* host_begin = (char* ) malloc(sizeof(char)*5);
            char* device_begin;
            cudaMalloc(&device_begin, sizeof(char)*5);
            cudaSetDevice(did);
            int device_id = 0;
            cudaGetDevice(&device_id);
            cout << "current device:" << device_id << endl;
            initialize<<<1,1>>>(device_begin);
            cudaMemcpy(host_begin, device_begin, 5, cudaMemcpyDeviceToHost);
            cout << did << host_begin << endl;
            free(host_begin);
            cudaFree(device_begin);

        }));
    }
    for (auto &thread: threads)
        thread.join ();
    return 0;
}