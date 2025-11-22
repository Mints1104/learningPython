import math 
from math import gcd
from collections import deque, Counter
import heapq
import time 
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


def bfs_shortest_path_grid(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0])

    # Directions: down, up, right, left
    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    queue = deque([start])
    visited = set([start])

    # distance map
    distance = {start: 0}

    # to reconstruct path
    parent = {start: None}

    while queue:
        row, col = queue.popleft()

        # If we've reached the goal, stop early
        if (row, col) == goal:
            break

        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc

            # Check boundaries
            if not (0 <= new_row < rows and 0 <= new_col < cols):
                continue

            # Skip walls
            if grid[new_row][new_col] == 1:
                continue

            # Skip visited cells
            if (new_row, new_col) in visited:
                continue

            visited.add((new_row, new_col))
            parent[(new_row, new_col)] = (row, col)
            distance[(new_row, new_col)] = distance[(row, col)] + 1
            queue.append((new_row, new_col))

    # If goal was never reached
    if goal not in parent:
        return None, None

    # Reconstruct path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()

    return path, distance[goal]

        
grid = [
    [0, 0, 0],
    [1, 1, 0],
    [0, 0, 0]
]

start = (0, 0)
goal = (2, 2)

path, dist = bfs_shortest_path_grid(grid, start, goal)
print("Path:", path)
print("Distance:", dist)


def bfs_me(grid, start, goal):
    queue = deque([start])
    visited = set([start])
    parent = {start: None}
    
    directions = [
        (1,0),
        (0,1),
        (-1,0),
        (0,-1)
    ]
    
    while queue:
        current_row, current_col = queue.popleft()
        
        for dr, dc in directions:
            neighbor_row = current_row + dr 
            neighbor_col = current_col + dc 
            
            #1. must be within boundaries
            
            if not(0 <= neighbor_row < len(grid) and 0 <= neighbor_col < len(grid[0])):
                continue 
            #2. we skip walls
            if grid[neighbor_row][neighbor_col] == 1:
                continue 
            #3. we skip anything visited
            if(neighbor_row, neighbor_col) in visited:
                continue
            
            visited.add((neighbor_row,neighbor_col))
            parent[(neighbor_row, neighbor_col)] = (current_row,current_col)
            queue.append((neighbor_row,neighbor_col))
        
    if goal not in parent:
        return None 
        
    path = []
    current = goal 
    while current is not None:
        path.append(current)
        current = parent[current]
            
    path.reverse()
    return path   
grid = [
    [0, 0, 0],
    [1, 1, 0],
    [0, 0, 0]
]

start = (0, 0)
goal = (2, 2) 
    
print(bfs_me(grid, start, goal))


def count_islands(grid):
    rows = len(grid)
    cols = len(grid[0])
    visited = set()

    # DFS to explore an entire island
    def dfs(row, col):
        if (
            row < 0 or row >= rows or
            col < 0 or col >= cols or
            grid[row][col] == 0 or
            (row, col) in visited
        ):
            return

        visited.add((row, col))

        dfs(row + 1, col)  # down
        dfs(row - 1, col)  # up
        dfs(row, col + 1)  # right
        dfs(row, col - 1)  # left

    island_count = 0

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1 and (row, col) not in visited:
                island_count += 1
                dfs(row, col)

    return island_count


grid = [
    [1,1,0,0,0],
    [1,0,0,1,1],
    [0,0,1,1,0],
    [0,0,0,0,1]
]

print(count_islands(grid))  # expected: 4

def is_valid_parentheses(s):
    stack = []
    matching_pair = {
        ')': '(',
        ']': '[',
        '}': '{'
    }
    
    for char in s:
        if char in "([{":
            stack.append(char)
            continue 
        
        if char in ")]}":
            if not stack:
                return False
            top = stack.pop()
            
            if matching_pair[char] != top:
                return False
    return len(stack) == 0

print(is_valid_parentheses("()"))        # True
print(is_valid_parentheses("([]{})"))    # True
print(is_valid_parentheses("([)]"))      # False
print(is_valid_parentheses("(((((]"))    # False
print(is_valid_parentheses("{[()]}"))    # True

def valid_brackets(s):
    stack = []
    
    matching_pairs = {
        ')': '(',
        ']': '[',
        '}': '{'
    }
    
    for char in s:
        if char in "([{":
            stack.append(char)
        if char in ")]}":
            if not stack:
                return False
            top = stack.pop()
            if matching_pairs[char] != top:
                return False
    return len(stack) == 0

print(valid_brackets("()"))        # True
print(valid_brackets("([]{})"))    # True
print(valid_brackets("([)]"))      # False
print(valid_brackets("(((((]"))    # False
print(valid_brackets("{[()]}"))    # True


def build_final_string(s):
    stack = []
    for char in s:
        if char == '#':
            if stack:
                stack.pop()
        else:
            stack.append(char)
    return "".join(stack)

def compare_strings(s,t):
    return build_final_string(s) == build_final_string(t)

def build_final(s):
    stack = []
    for char in s:
        if char == '#':
            if stack:
                stack.pop()
        else:
            stack.append(char)
            
    return "".join(stack)

def compare_strings_me(s,t):
    return build_final(s) == build_final(t)
print("--------------------------------------------------")
print(compare_strings_me("ab#c", "ad#c"))   # True
print(compare_strings_me("ab##", "c#d#"))   # True
print(compare_strings_me("a#c", "b"))       # False
print(compare_strings_me("xy##", "z#"))     # True



def find_smallest_window_patterns(patterns,text):
    
#1. collect all occurrences
    occurrences = [] # start,end,pattern_id
    
    for pattern_id, pattern in enumerate(patterns):
        search_index = 0
        while True:
            pos = text.find(pattern,search_index)
            if pos == -1:
                return [-1,-1]
            occurrences.append((pos,pos+len(pattern) -1, pattern_id))
            search_index = pos + 1
            
    #2. sort all occurrences by start index
    occurrences.sort(key=lambda x: x[0])
    
    required_pattern_count = len(patterns)
    pattern_counter = {i: 0 for i in range(required_pattern_count)}
    covered = 0
    best_start = -1
    best_end = -1
    smallest_length = float("inf")
    left = 0
    
    for right in range(len(occurrences)):
        start_r, end_r, pat_r = occurrences[right]
        
        if pattern_counter[pat_r] == 0:
            covered += 1
        pattern_counter[pat_r] += 1
        
        while covered == required_pattern_count:
            start_l, end_l, pat_l = occurrences[left]
            
            current_window_start = start_l 
            current_window_end = max(
                end 
                for (_, end, _) in occurrences[left:right+1]
                )
            current_length = current_window_end - current_window_start + 1
            
            if current_length < smallest_length:
                smallest_length = current_length
                best_start = current_window_start
                best_end = current_window_end
        
            pattern_counter[pat_l] -=1
            if pattern_counter[pat_l] == 0:
                covered -=1
            left +=1
            
    return [best_start,best_end]
            
            
def minimum_window_substring(s,t):
    if t == "": return ""
    
    countT,window = {},{}
    
    for c in t:
        countT[c] = 1 + countT.get(c,0)
    
    have, need = 0,len(countT)
    res, resLen = [-1,-1], float("inf")
    l = 0
    for r in range(len(s)):
        c = s[r]
        window[c] = 1 + window.get(c,0)
        
        if c in countT and window[c] == countT:
            have +=1
        while have == need:
            # update our result 
            if(r - l + 1) < resLen:
                res = [l,r]
                resLen = (r - l + 1)
            # pop from the left of our window
            window[s[l]] -= 1
            if s[l] in countT and window[s[l]] < countT[s[l]]:
                have -=1
            l+=1
            
    l,r = res 
    return s[l:r+1] if resLen != float("inf") else ""        

def threesum(nums):
    nums.sort()
    results = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left = i + 1 
        right = len(nums) - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum < 0:
                left +=1
            elif current_sum > 0:
                right -=1
            else:
                results.append([nums[i]],nums[left],nums[right])
                left +=1
                right -=1
            while left < right and nums[left] == nums[left-1]:
                left +=1
            while left < right and nums[right] == nums[right+1]:
                right -=1
                
    return results


def maxArea(height):
    left = 0
    right = len(height) - 1
    max_area = 0
    while left < right:
        h = min(height[left], height[right])
        width = right - left
        area = h * width
        max_area = max(max_area,area)
        if height[left] < height[right]:
            left +=1
        else:
            right -=1
            
    return max_area

heights = [1,8,6,2,5,4,8,3,7]

print(maxArea(heights))

def three_sum_closest(nums,target):
    nums.sort()
    closest_sum = float('inf')
    
    for i in range(len(nums) - 2):
        left = i + 1
        right = len(nums) - 1
        
        while left < right:
            current = nums[i] + nums[left] + nums[right]
            
           
            if abs(current - target) < abs(closest_sum - target):
                closest_sum = current
    
            
            
            if current < target:
                left +=1
            else:
                right -=1
    return closest_sum


moar_nums = [-1,2,1,-4]
print(three_sum_closest(moar_nums,1))


def reorderList(head):
        if not head or not head.next:
            return

        # 1. Find middle of the list
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # slow now points to the middle node

        # 2. Reverse the second half
        prev = None
        current = slow.next
        slow.next = None  # split into two lists

        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node

        second_half_head = prev  # head of reversed second half

        # 3. Merge both halves alternately
        first = head
        second = second_half_head

        while second:
            temp1 = first.next
            temp2 = second.next

            first.next = second
            second.next = temp1

            first = temp1
            second = temp2
            
            
def four_sum(nums,target):
    nums.sort()
    results = []
    n = len(nums)
    
    for i in range(n-3):
        for j in range(i + 1, n- 2):
            left = j + 1
            right = n - 1 
            while left < right:
                
                total  = nums[left] + nums[right] + nums[i] + nums[j]
                if total < target:
                    left +=1
                elif total > target:
                    right -=1
                else:
                    results.append([nums[i],nums[j],nums[left],nums[right]])
                    left +=1
                    right -=1
    
    return results

            
          
            
four_sum_arr = [1,0,-1,0,-2,2]
target = 0

print(four_sum(four_sum_arr,0))
            
def length_of_longest_substring(s):
    char_set = set()
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left +=1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
        
    return max_len


print(length_of_longest_substring("abcabcbb"))

def character_replacement(s,k):
    freq = {}
    left = 0
    max_len = 0
    max_count = 0
    
    for right in range(len(s)):
        char = s[right]
        freq[char] = freq.get(char,0) + 1
        
        max_count = max(max_count,freq[char])
        window_size = right - left + 1
        while (window_size - max_count) > k:
            freq[s[left]] -=1
            left +=1
            
        max_len = max(max_len, right - left + 1)
        
    return max_len
       
       


def climbStairs(n):
    if n <=2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range (3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
        
    return dp[n]

def findKthLargest(nums,k):
    
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap,num)
        
        # if the heap is now larger than k, remove smallest element
        
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    
    return min_heap[0]


def time_decorator(func):
    
    def wrapper():
        start_time = time.time()
        func()
        end_time = time.time()
        print(f"Function took {end_time - start_time:.4f} seconds")
    return wrapper

@time_decorator
def slow_function():
    time.sleep(1)
    print("Function complete!")
    
slow_function()

def debug(func):
    def wrapper(*args, **kwargs):
        
        print(f"Calling {func.__name__} with args: {args}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result 
    return wrapper

@debug 
def add(a,b):
    return a + b

@debug 
def greet(name):
    return f"Hello, {name}!"

add(5,3)
greet("Alice")


def max_avg_subarray(nums,k):
    
    current_sum = sum(nums[0:k])
    current_max = current_sum
    left = 0
    
    for right in range(k,len(nums)):
        current_sum += nums[right]
        current_sum -= nums[left]
        left +=1
        
        current_max = max(current_max,current_sum)
        
            
        
def reverse_vowels_of_string(s):
    vowels = set("aeiouAEIOU")
    s = list(s)
    left = 0
    right = len(s) - 1
   
    
    while left < right:
        left_char = s[left]
        right_char = s[right]
        
        if (left_char in vowels) and (right_char in vowels):
            s[left], s[right] = s[right], s[left]
            left +=1
            right -=1
            
        elif(left_char in vowels):
            right -=1
        elif (right in vowels):
            left +=1
        else:
            
            
            left+=1
            right -=1
        
    return "".join(s)


print(reverse_vowels_of_string("IceCreAm"))        


def gcd_strings(s,t):
    if s + t != t + s:
        return ""
    
    gcd_length = gcd(len(s),len(t))
    
    return s[:gcd_length]


def plant_flowers(flowerBed,n):
    length = len(flowerBed)
    
    for i in range(length):
        if i == 0:
            if flowerBed[i] == 0 and flowerBed[i+1] == 0:
                n -=1
                flowerBed[i] = 1
        elif i == length - 1:
            if flowerBed[i] == 0 and flowerBed[i -1] == 0:
                n -=1
                flowerBed[i] = 1
        else:
            if flowerBed[i] == 0 and flowerBed[i-1] == 0 and flowerBed[i+1] == 0:
                n -=1
                flowerBed[i] = 1
        
        if n<=0:
            return True
        
    return n<=0 

def plant_flower_optimised(flowerbed,n):
    length = len(flowerbed)
    
    for i in range(length):
        left_empty = (i == 0) or (flowerbed[i-1] == 0)
        right_empty = (i == length -1) or (flowerbed[i+1] == 0)
        center_empty = (flowerbed[i] == 0)
        
        if left_empty and center_empty and right_empty:
            flowerbed[i] = 1
            n -=1
            if n == 0:
                return True
    return n <= 0 


def reverseWords(s):
    left = 0
    right = len(s) - 1
    words = s.split()
    
    while left < right:
        left_word = s[left]
        right_word = s[right]
        
        s[left], s[right] = s[right], s[left]
        left +=1
        right -=1
        
        
    return "".join(s)

def findDifference(nums1,nums2):
    nums2_set = set(nums2)
    nums1_set = set(nums1)
    
    for num in nums1:
        if num in nums2_set:
            nums1.remove(num)
            
    for num in nums2:
        if num in nums1_set:
            nums2.remove(num)
            
    return [nums1,nums2]
            
def isSubsequence(s,t):
    str1_pointer = 0
    str2_pointer = 0
       
    while (str1_pointer < len(s)) and (str2_pointer < len(t)):
        str1_char = s[str1_pointer]
        str2_char = t[str2_pointer]
        
        if str1_char == str2_char:
            str1_pointer +=1
            
        str2_pointer +=1
        
    return str1_pointer == len(s)


def closeStrings(word1,word2):
    
    if set(word1) != set(word2):
        return False
    
    freq1 = Counter(word1)
    freq2 = Counter(word2)