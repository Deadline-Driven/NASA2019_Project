#include <chrono>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <android/NeuralNetworks.h>
#include "NeuroPilotTFLiteShim.h"

using namespace std;

int main(int argc, char** argv) {
    // Make compiler happy( unused variable warning)
    (void)(argc);
    (void)(argv);

    // Model path
    const char* model_path = "/data/local/tmp/nasa_srmodel.tflite";
    const char* input_path = "/data/local/tmp/target.bin";
    const char* output_path = "/data/local/tmp/result.bin";

    ANeuralNetworksTFLite *tflite = nullptr;
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<double> elapsed;

    do {
        // Create TFLite instance with a model file path
        if ( ANeuroPilotTFLiteWrapper_makeTFLite(&tflite, model_path) !=
                ANEURALNETWORKS_NO_ERROR) {
                    cout << "Fail to create TFLite instance" << endl;
                    break;
                }
        
        ANeuroPilotTFLiteWrapper_setAllowFp16PrecisionForFp32(tflite, true);

        // Get input tensor
        TFLiteTensorExt inputTensor;
        if (ANeuroPilotTFLiteWrapper_getTensorByIndex(tflite,
                TFLITE_BUFFER_TYPE_INPUT,
                &inputTensor,
                0) != ANEURALNETWORKS_NO_ERROR) {
            cout << "Fail to get input tensor" << endl;
            break;
        }

        // Print the input tensor information
        cout << "Input tensor information" << endl;
        cout << "type: " << inputTensor.type << endl;
        cout << "dimsSize: " << inputTensor.dimsSize << endl;

        for (int i = 0 ; i < inputTensor.dimsSize ; i++) {
            cout << inputTensor.dims[i] << " ";
        }

        cout << endl;
        cout << "Size of input tensor: " << inputTensor.bufferSize << endl;

        // Fill data to the buffer of the input tensor
        std::ifstream input_fs(input_path);

        if (!input_fs.good()) {
            cout << "Fail to read " << input_path;
            break;
        }

        // Fill data to the buffer of the input tensor
        input_fs.read((char*)inputTensor.buffer, inputTensor.bufferSize);
        input_fs.close();

        start_time = std::chrono::high_resolution_clock::now();

        // Invoke
        if (ANeuroPilotTFLiteWrapper_invoke(tflite) != ANEURALNETWORKS_NO_ERROR) {
            cout << "Fail to invoke" << endl;
            break;
        }

        end_time = std::chrono::high_resolution_clock::now();

        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

        // Get output buffer
        TFLiteTensorExt outputTensor;

        if (ANeuroPilotTFLiteWrapper_getTensorByIndex(tflite,
                TFLITE_BUFFER_TYPE_OUTPUT,
                &outputTensor,
                0) != ANEURALNETWORKS_NO_ERROR) {
            cout << "Fail to get output tensor" << endl;
            break;
        }

        // Print the output tensor information
        cout << "Output tensor information" << endl;
        cout << "type: " << outputTensor.type << endl;
        cout << "dimsSize: " << outputTensor.dimsSize << endl;

        for (int i = 0 ; i < outputTensor.dimsSize ; i++) {
            cout << outputTensor.dims[i] << " ";
        }

        cout << endl;
        cout << "Size of output tensor: " << outputTensor.bufferSize << endl;

        cout << "Inference time : " << elapsed.count() * 1000 << " ms" << endl;

        std::ofstream output_fs(output_path);
        
        if (!output_fs.good()) {
            cout << "Fail to read " << output_path;
            break;
        }

        output_fs.write((char*)outputTensor.buffer, outputTensor.bufferSize);
        output_fs.close();

    } while(false);

    if ( tflite != nullptr) {
        ANeuroPilotTFLiteWrapper_free(tflite);
    }

    return 0;
}