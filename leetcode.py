def contains_duplicate(nums,k):
    indices = {}
    for index,num in enumerate(nums):
        if num not in indices:
            print(f"Adding num: {num} at index: {index} to indices")
            indices[num] = index
        else:
           stored_index = indices[num]
           print(f"Stored Index: {stored_index} for num: {num}")
           print(f"Current Index, Num: {index},{num}")
           test = abs(index - stored_index) + 1
           print(f"Result: {test}")
           if abs(index - stored_index) <= k:
               print(f"Index: {index} - Stored Index {stored_index}")
               return True
           indices[num] = index
        return False
arr = [1,2,3,1,2,3]
print(f"Array: {arr}")           
print(contains_duplicate(arr,2)) 

arr2 = [1,5,10,9,64,29,100,45,90,200]

def find_maximum(arr):
    max_elem = 0
    for num in arr:
        if num > max_elem:
            max_elem = num
            
    return max_elem

print(find_maximum(arr2))

def naive_two_sum(nums,k):
    
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == k:
                return nums[i],nums[j]
    return None

def two_sum(nums,k):
    seen = {}
    for num in nums:
        if k - num in seen:
            return num, k - num
        seen[num] = True
    return None
    

arr3 = [1,5,7]
print(naive_two_sum(arr2,14))
print("h")
print(two_sum(arr2,14))
      
arr4 = [1,3,5,8,12,15,17]  
        
def two_sum_sorted(nums,k):
    left = 0
    right = len(nums) - 1
    while left < right:
        if nums[left] + nums[right] == k:
            return nums[left],nums[right]
        elif nums[left] + nums[right] > k:
            right -=1
        elif nums[left] + nums[right] < k:
            left +=1
       
        
    return None

print(two_sum_sorted(arr4,20))

def length_of_longest_substring_me(s):
    left = 0
    right = 1
    current_max = 0
    chars = {}
    chars[s[left]] = True
    while right < len(s):
        print(f"Checking: {s[right]}")
        if s[right] not in chars:
            print(f"Adding: {s[right]} to chars")
            chars[s[right]] = True
        else:
            print(f"Char: {s[right]} already in set")
            print(f"Current Max: {current_max}")
            print(f"Popping {s[left]}")
            chars.pop(s[left],None)
            chars[s[right]] = True
            print(f"Current Set: {chars.keys()}")

            left +=1
        right +=1
        current_max = max(current_max,len(chars))
    print(chars.keys())
    return current_max

def length_of_longest_substring(s):
    left = 0
    chars = set()
    current_max = 0
    for right in range(len(s)):
        while s[right] in chars:
            chars.remove(s[left])
            left +=1
        chars.add(s[right])
        current_max = max(current_max, right - left + 1)

print(length_of_longest_substring_me("abba"))

people = [
    ("Alice", 25),
    ("Bob", 19),
    ("Charlie", 25),
    ("Diana", 22),
    ("Eve", 19),
    ("Frank", 30)
]

print(people[0][0])


    