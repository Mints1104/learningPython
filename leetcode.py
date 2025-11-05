import math 
from collections import deque

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

print(people[1][1])
#  s[left], s[right] = s[right], s[left]
def sort_people(people):
    mid_point = len(people) / 2 + 1
    left = 0
    right = len(people) - 1
    
   # for name, age in enumerate(people):
   
    while left < len(people) - 1:
        left_age = people[left][1]
        right_age = people[right][1]
        print(f"Left Age: {left_age}")
        print(f"Right Age: {right_age}")
        counter = 0
        
        if right_age < left_age:
            print("Right less than left, move left up")
            left_age, right_age = right_age, left_age
            left +=1
            counter +=1
        elif left_age < right_age and right != mid_point:
            print("Left less than right, move right down")
            right -=1
            counter +=1
            print(f"Right is now: {right}")
        break

    return people

def bubble_sort(people):
    n = len(people)

    for _ in range(n - 1):  # repeat passes through the list
        for i in range(n - 1):
            person_a = people[i]
            person_b = people[i + 1]

            # compare by age first, then by name alphabetically
            age_a, name_a = person_a[1], person_a[0]
            age_b, name_b = person_b[1], person_b[0]

            # if the left person should come after the right person, swap them
            if age_a > age_b or (age_a == age_b and name_a > name_b):
                people[i], people[i + 1] = person_b, person_a

    return people
print(bubble_sort(people))

def binary_search(nums,target):
    left = 0
    right = len(nums) - 1
    
    while left <= right:
        mid = (left + right) //2
        mid_value = nums[mid]
        if mid_value == target:
            return mid
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1


def lower_bound(nums,target):
    left,right = 0, len(nums)
    while left< right:
        mid = (left + right) //2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

def upper_bound(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left  # first index where nums[index] > target


more_nums = [1,3,4,6,8,10,12,14,17,20]
print(binary_search(more_nums, 10))  # should return 5
print(binary_search(more_nums, 7))   # should return -1
print(binary_search(more_nums, 1))   # should return 0
print(binary_search(more_nums, 20))  # should return 9
        

def my_binary_search(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = (left + right) //2
        
        if nums[mid] == target:
            return mid 
        
        if nums[mid] > target:
            right = mid -1
        else:
            left = mid + 1
            
    return -1

print(my_binary_search(more_nums,10))
            
            
def can_finish_in_time(piles, hours_available, eating_speed):
    total_hours_needed = 0
    for pile in piles:
        total_hours_needed += math.ceil(pile/eating_speed)
        
    return total_hours_needed <= hours_available
        
def find_minimum_eating_speed(piles, hours_available):
    left_speed = 1
    right_speed = max(piles)
    
    while left_speed < right_speed:
        middle_speed = (left_speed + right_speed) // 2
        
        if can_finish_in_time(piles,hours_available,middle_speed):
            right_speed = middle_speed
        else:
            left_speed = middle_speed + 1
            
    return left_speed

piles = [3, 6, 7, 11]
hours_available = 8

minimum_speed = find_minimum_eating_speed(piles, hours_available)
print(minimum_speed)  
        
        
        
graph = {
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A", "E"],
    "D": ["B"],
    "E": ["C"]
}

print("----------------------------------------------------------------------------------")

def dfs(node, visited, graph):
    if node in visited:
        return
    visited.add(node)
    print(node)
    for neighbor in graph[node]:
        print(f"neighbor of {node}: {neighbor}")
        dfs(neighbor,visited,graph)
        
def bfs(start_node, graph):
    visited = set([start_node])
    queue = deque([start_node])
    
    while queue:
        node = queue.popleft()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                print(f"adding: {neighbor} to visited")
                visited.add(neighbor)
                queue.append(neighbor)
    
visited = set()
dfs("A",visited,graph)
bfs("A",graph)

def bfs_shortest_path(start_node,target_node,graph):
    visited = set([start_node])
    parent = {start_node: None}
    queue = deque([start_node])
    
    while queue:
        node = queue.popleft()
        
        # Stop early if we found our target
        if node == target_node:
            break
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node 
                queue.append(neighbor)
        
        if target_node not in parent:
            return None
        
        #reconstruct
        path = []
        current = target_node
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path
   
path = bfs_shortest_path("A", "E", graph)
print(path)

