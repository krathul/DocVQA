from fuzzywuzzy import fuzz

def filter_and_replace(text, keywords):
      for keyword in keywords:
        #if text.isdigit():
          #pass
        if fuzz.ratio(keyword, text) >= 90:
          return keyword
        else:
          return text

def remove_similar_keywords(keyword_list):
      new_list = []
      for i, keyword in enumerate(keyword_list):
          is_similar = False
          for j in range(i + 1, len(keyword_list)):
              if fuzz.token_set_ratio(keyword, keyword_list[j]) >= 80:
                  is_similar = True
                  break
          if not is_similar:
              new_list.append(keyword)
      return new_list

def lowercase_except_digits_floats(lst):
      new_lst = []
      for string in lst:
          new_string = ''
          for char in string:
              if char.isnumeric() or char.isdecimal() or char == '.':
                  new_string += char
              else:
                  new_string += char.lower()
          new_lst.append(new_string)
      return new_lst

def fuzzy_match(keyword, keyword_list):
      for kw in keyword_list:
          if fuzz.ratio(keyword, kw) > 80:
              return True
      return False
