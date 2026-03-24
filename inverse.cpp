# include <iostream>

void inverse(unsigned char *a, int height, int width) {
    int total_bytes = height * width * 4;
    for (size_t i = 0; i< total_bytes; i+=4){
        a[i] = 255 - a[i];
        a[i+1] = 255 - a[i+1];
        a[i+2] = 255 - a[i+2]; 
    }
}

void printImage(unsigned char* data, int width, int height) {
    std::cout << "[";
    int totalBytes = width * height * 4;
    for (int i = 0; i < totalBytes; ++i) {
        std::cout << (int)data[i];
        if (i < totalBytes - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main(){
    int width = 2;
    int height = 2;
    unsigned char image[16] = {0, 0, 0, 255, 255, 255, 255, 255, 128, 128, 128, 255, 64, 64, 64, 255};
    std::cout << "Original Image: ";
    printImage(image, width, height);
    inverse(image, height, width);
    std::cout << "Inverted Image: ";
    printImage(image, width, height);
    return 0;
}