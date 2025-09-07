---
title: Generic Pointers
draft: true
tags:
  - C
---


Generic pointers (`void*`) are one of C's most useful features for writing flexible code. They allow you to create functions and data structures that work with any data type. Let's explore this concept with simple, practical examples.

## What is a Generic Pointer?
A `void*` (void pointer) is a pointer that can point to any data type. Unlike regular pointers like `int*` or `char*`, it doesn't know what type of data it's pointing to.

```c
#include <stdio.h>

int main() {
    int number = 42;
    char letter = 'A';
    double pi = 3.14;

    void* generic_ptr;  // Can point to anything

    generic_ptr = &number;  // Points to int
    printf("Integer: %d\n", *(int*)generic_ptr);

    generic_ptr = &letter;  // Points to char
    printf("Character: %c\n", *(char*)generic_ptr);

    generic_ptr = &pi;      // Points to double
    printf("Double: %.2f\n", *(double*)generic_ptr);

    return 0;
}
```

## Key Rules

1. **Cannot dereference directly** - Must cast to specific type first
2. **Cannot do pointer arithmetic** - No `ptr++` or `ptr + 1`
3. **Must know the original type** when casting back

## Simple Use Case 1: Generic Swap Function

Instead of writing separate swap functions for each type, write one generic version:

```c
#include <stdio.h>
#include <string.h>

// Generic swap function
void swap(void* a, void* b, size_t size) {
    // Create temporary buffer
    char temp[size];

    // Copy a to temp, b to a, temp to b
    memcpy(temp, a, size);
    memcpy(a, b, size);
    memcpy(b, temp, size);
}

int main() {
    // Swap integers
    int x = 10, y = 20;
    printf("Before: x=%d, y=%d\n", x, y);
    swap(&x, &y, sizeof(int));
    printf("After: x=%d, y=%d\n\n", x, y);

    // Swap doubles
    double a = 3.14, b = 2.71;
    printf("Before: a=%.2f, b=%.2f\n", a, b);
    swap(&a, &b, sizeof(double));
    printf("After: a=%.2f, b=%.2f\n\n", a, b);

    // Swap characters
    char c1 = 'X', c2 = 'Y';
    printf("Before: c1=%c, c2=%c\n", c1, c2);
    swap(&c1, &c2, sizeof(char));
    printf("After: c1=%c, c2=%c\n", c1, c2);

    return 0;
}
```

## Simple Use Case 2: Generic Array

Create a simple array that can store any type:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple generic array
typedef struct {
    void* data;         // Pointer to data
    size_t element_size; // Size of each element
    size_t count;       // Number of elements
    size_t capacity;    // Maximum elements
} GenericArray;

// Create new array
GenericArray* array_create(size_t element_size, size_t initial_capacity) {
    GenericArray* arr = malloc(sizeof(GenericArray));
    arr->data = malloc(element_size * initial_capacity);
    arr->element_size = element_size;
    arr->count = 0;
    arr->capacity = initial_capacity;
    return arr;
}

// Add element to array
void array_add(GenericArray* arr, void* element) {
    if (arr->count < arr->capacity) {
        // Calculate position and copy element
        char* dest = (char*)arr->data + (arr->count * arr->element_size);
        memcpy(dest, element, arr->element_size);
        arr->count++;
    }
}

// Get element from array
void* array_get(GenericArray* arr, size_t index) {
    if (index < arr->count) {
        return (char*)arr->data + (index * arr->element_size);
    }
    return NULL;
}

// Free array
void array_free(GenericArray* arr) {
    free(arr->data);
    free(arr);
}

int main() {
    // Create array for integers
    GenericArray* int_array = array_create(sizeof(int), 5);

    // Add some integers
    int values[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++) {
        array_add(int_array, &values[i]);
    }

    // Print integers
    printf("Integers: ");
    for (size_t i = 0; i < int_array->count; i++) {
        int* value = (int*)array_get(int_array, i);
        printf("%d ", *value);
    }
    printf("\n");

    array_free(int_array);

    // Create array for doubles
    GenericArray* double_array = array_create(sizeof(double), 3);

    double decimals[] = {1.1, 2.2, 3.3};
    for (int i = 0; i < 3; i++) {
        array_add(double_array, &decimals[i]);
    }

    printf("Doubles: ");
    for (size_t i = 0; i < double_array->count; i++) {
        double* value = (double*)array_get(double_array, i);
        printf("%.1f ", *value);
    }
    printf("\n");

    array_free(double_array);

    return 0;
}
```

## Simple Use Case 3: Generic Print Function

Create a function that can print different types:

```c
#include <stdio.h>

// Enum to identify data types
typedef enum {
    TYPE_INT,
    TYPE_DOUBLE,
    TYPE_CHAR,
    TYPE_STRING
} DataType;

// Generic print function
void generic_print(void* data, DataType type) {
    switch (type) {
        case TYPE_INT:
            printf("Integer: %d\n", *(int*)data);
            break;
        case TYPE_DOUBLE:
            printf("Double: %.2f\n", *(double*)data);
            break;
        case TYPE_CHAR:
            printf("Character: %c\n", *(char*)data);
            break;
        case TYPE_STRING:
            printf("String: %s\n", (char*)data);
            break;
    }
}

int main() {
    int num = 42;
    double pi = 3.14159;
    char letter = 'A';
    char* message = "Hello, World!";

    generic_print(&num, TYPE_INT);
    generic_print(&pi, TYPE_DOUBLE);
    generic_print(&letter, TYPE_CHAR);
    generic_print(message, TYPE_STRING);

    return 0;
}
```

## Conclusion

Generic pointers (`void*`) are a powerful tool in C for writing flexible, reusable code. While they require careful handling and lose compile-time type safety, they enable you to create elegant solutions that work with any data type. The key is understanding that `void*` is just an address - it's up to you to remember and correctly handle what's stored at that address.
