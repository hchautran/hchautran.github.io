---
title: Pointers in C 
draft: false
tags:
  - C
  - notes
---

Pointers are one of C's most powerful and misunderstood features. This guide walks through the essentials: what pointers are, how to use them safely, and common pitfalls.

## What is a pointer?

A pointer stores the address of another object (variable, function, or dynamically allocated memory).

```c
#include <stdio.h>

int main(void) {
    int x = 42;          // a normal integer
    int *p = &x;         // p points to x

    printf("x=%d\n", x);          // value
    printf("&x=%p\n", (void*)&x); // address of x
    printf("p=%p\n", (void*)p);   // pointer value (address stored)
    printf("*p=%d\n", *p);        // dereference → value at address
}
```

Key ideas:
- `&x` gives the address of `x`.
- `*p` dereferences `p` to access the object it points to.
- The type `int*` means “pointer to int.” Types must match for correct dereferencing.

## Pointer types and `const`

```c
int *       p;   // pointer to       int (modifiable through p)
const int * pc;  // pointer to const int (cannot modify *pc through pc)
int * const cp = &x; // const pointer to int (cp cannot change, *cp can)
const int * const cpc = &x; // const pointer to const int
```

- “const to the left of the star” protects the pointee; “const to the right” fixes the pointer.

## Pointer arithmetic (with arrays)

In C, arrays decay to pointers to their first element in many expressions. Pointer arithmetic moves in units of the pointed-to type.

```c
#include <stdio.h>

int main(void) {
    int a[5] = {10, 20, 30, 40, 50};
    int *p = a;           // same as &a[0]

    printf("*p=%d\n", *p);       // 10
    printf("*(p+2)=%d\n", *(p+2)); // 30

    for (int i = 0; i < 5; i++) {
        printf("%d ", *(p + i));
    }
    printf("\n");
}
```

Rules:
- You may move within the same array (`p`, `p+1`, …, `p+n`).
- Going outside the array (except the one-past-the-end position for comparisons) is undefined behavior.

## Pointers and strings

Strings in C are arrays of `char` terminated by `\0`.

```c
#include <stdio.h>

int main(void) {
    char s[] = "hello";   // array with space for '\0'
    char *ps = s;          // points to first character

    for (; *ps != '\0'; ps++) {
        putchar(*ps);
    }
    putchar('\n');
}
```

Common mistakes:
- Forgetting the terminating `\0` when building strings manually.
- Modifying string literals (e.g., `char *s = "hi"; s[0] = 'H';` is undefined).

## Pointers to pointers

Useful for returning allocated memory or modifying caller-owned pointers.

```c
#include <stdlib.h>
#include <stdbool.h>

bool make_buffer(size_t n, int **out) {
    int *buf = malloc(n * sizeof *buf);
    if (!buf) return false;
    *out = buf;   // set caller's pointer
    return true;
}
```

Usage:
```c
int *data = NULL;
if (make_buffer(100, &data)) {
    // use data[0..99]
    free(data);
}
```

## Function pointers

Store addresses of functions to implement callbacks or pluggable behavior.

```c
#include <stdio.h>

int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }

int apply(int (*op)(int,int), int x, int y) { return op(x, y); }

int main(void) {
    printf("%d\n", apply(add, 3, 4)); // 7
    printf("%d\n", apply(mul, 3, 4)); // 12
}
```

## Dynamic memory and ownership

```c
#include <stdlib.h>

int *make_array(size_t n) {
    int *p = malloc(n * sizeof *p);
    if (!p) return NULL;  // check for allocation failure
    return p;              // caller owns and must free
}

int main(void) {
    size_t n = 1000;
    int *arr = make_array(n);
    if (!arr) return 1;
    // use arr[0..n-1]
    free(arr);
}
```

Guidelines:
- Every successful `malloc/calloc/realloc` must have a matching `free`.
- After `free(p)`, set `p = NULL;` to avoid dangling pointers.
- Avoid returning pointers to local (stack) variables.

## Common pitfalls (and how to avoid them)

1. Dangling pointers
   - Cause: using memory after it’s freed or out of scope.
   - Fix: `free` once; set to `NULL`; never return address of locals.

2. Out-of-bounds access
   - Cause: incorrect pointer arithmetic or loop bounds.
   - Fix: keep size variables; prefer indexing; validate boundaries.

3. Incorrect `const` usage
   - Cause: accidentally modifying through a pointer meant to be read-only.
   - Fix: use `const` for pointees you do not intend to modify.

4. Mixing types
   - Cause: casting to wrong type and dereferencing.
   - Fix: keep types consistent; only cast when necessary and correct.

5. Memory leaks
   - Cause: losing the only pointer to allocated memory.
   - Fix: define clear ownership; pair allocations with frees on all paths.

## Debugging tips

- Use sanitizers (`-fsanitize=address,undefined`) and warnings (`-Wall -Wextra -Wpedantic`).
- Initialize pointers to `NULL` and check before dereferencing.
- In complex code, document ownership and lifetime in comments.

## See also

- [Initialization in C](Initialization%20in%20C)

## Summary

- Pointers store addresses; `*` dereferences; `&` takes addresses.
- Pointer arithmetic works in element-sized steps and stays within arrays.
- `const` on the left protects the pointee; on the right fixes the pointer.
- Use pointers for dynamic memory, indirection (pointer-to-pointer), and callbacks (function pointers).
- Avoid UB: no out-of-bounds, no dangling pointers, match `malloc` with `free`.


