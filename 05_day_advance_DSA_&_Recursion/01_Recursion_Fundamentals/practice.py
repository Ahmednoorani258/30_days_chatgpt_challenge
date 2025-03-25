# after learning fundamentals trying writing factorial and fibanoci by yourself

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

arr = [1,2,3,4,5,6,7,8,9,10]
target = 5
start = 0
end = len(arr) - 1


def binary_search(arr,target,start,end):
    if start > end:
        return -1
    mid = (start + end) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        binary_search(arr,target,mid+1,end)
    else:
        binary_search(arr,target,start,mid-1)
        
print(factorial(5))
print(fibonacci(5))
print(binary_search(arr,target,start,end))
