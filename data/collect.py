# import praw
# import pandas as pd
# import datetime
# import re

# # Authentication setup
# with open('../credentials.txt', 'r') as file:
#     secret, client_id = file.read().split(';')
# user_agent = "windows:tr.DILBERT:v1.0 (by /u/imemmul)"

# reddit = praw.Reddit(
#     client_id=client_id,
#     client_secret=secret,
#     user_agent=user_agent,
# )

# subreddits = ['Turkey', 'turkish', 'TurkeyJerky']
# combined_data = []

# controversial_keywords = [
#     'mülteci', 'suriyeli', 'göçmen', 'ermeni', 'kürt', 
#     'yunan', 'azınlık', 'siyaset', 'iktidar', 'muhalefet', 'tayyip'
# ]

# hate_keywords = [
#     'siktir', 'amcık', 'am', 'göt', 'orospu', 'piç', 'yavşak', 'ibne', 
#     'pezevenk', 'yarrak', 'çük', 'bok', 'gerizekalı', 'salak'
# ]

# def contains_controversial_keywords(text):
#     if not text:
#         return False
#     text = text.lower()
#     for keyword in controversial_keywords:
#         if re.search(r'\b' + re.escape(keyword) + r'\b', text):
#             return True
#     return False

# def check_profanity(text):
#     if not text:
#         return False
    
#     text = text.lower()
    
#     normalized_text = re.sub(r'[-_.,:;*`\'^+~\\/#|\s]', '', text)
    
#     for word in hate_keywords:
#         pattern = ''
#         for char in word:
#             if char == 'a':
#                 pattern += '[a@4]'
#             elif char == 'i':
#                 pattern += '[i1!]'
#             elif char == 'o':
#                 pattern += '[o0]'
#             elif char == 'e':
#                 pattern += '[e3]'
#             elif char == 's':
#                 pattern += '[s5$]'
#             elif char == 'u':
#                 pattern += '[uü]'
#             elif char == 'c':
#                 pattern += '[cç]'
#             elif char == 'g':
#                 pattern += '[gğ]'
#             else:
#                 pattern += char
                
#         pattern = ''.join([c + '+' if i > 0 and c == pattern[i-1] else c for i, c in enumerate(pattern)])
        
#         if re.search(pattern, normalized_text):
#             return True
            
#     return False

# post_count = 0
# keyword_matched_count = 0
# profanity_matched_count = 0

# for subreddit_name in subreddits:
#     print(f"Processing subreddit: {subreddit_name}")
#     subreddit = reddit.subreddit(subreddit_name)
    
#     for sort_method in ['hot', 'top']:
#         print(f"  Using sort method: {sort_method}")
#         if sort_method in ['top']:
#             posts_method = getattr(subreddit, sort_method)(limit=100, time_filter='all')
#         else:
#             posts_method = getattr(subreddit, sort_method)(limit=100)

#         for submission in posts_method:
#             post_count += 1
            
#             if not submission.selftext and not submission.title:
#                 continue
            
#             title_contains_keywords = contains_controversial_keywords(submission.title)
#             body_contains_keywords = contains_controversial_keywords(submission.selftext)
            
#             # Check if title or body contains profanity
#             title_contains_profanity = check_profanity(submission.title)
#             body_contains_profanity = check_profanity(submission.selftext)
            
#             # Track if post contains either controversial keywords or profanity
#             contains_keywords = title_contains_keywords or body_contains_keywords
#             contains_profanity = title_contains_profanity or body_contains_profanity
            
#             if contains_keywords:
#                 keyword_matched_count += 1
            
#             if contains_profanity:
#                 profanity_matched_count += 1
                
            
#             combined_data.append({
#                 'id': submission.id,
#                 'subreddit': subreddit_name,
#                 'title': submission.title,
#                 'body': submission.selftext,
#                 'created_utc': datetime.datetime.fromtimestamp(submission.created_utc),
#                 'contains_keywords': contains_keywords,
#                 'contains_profanity': contains_profanity,
#                 'keywords_in_title': title_contains_keywords,
#                 'keywords_in_body': body_contains_keywords,
#                 'profanity_in_title': title_contains_profanity,
#                 'profanity_in_body': body_contains_profanity
#             })
            
#             if post_count % 10 == 0:
#                 print(f"Processed {post_count} posts, found {keyword_matched_count} with keywords, {profanity_matched_count} with profanity")

# df = pd.DataFrame(combined_data)

# df.to_excel('turkish_reddit_data_full_2.xlsx', index=False)
# df.to_csv('turkish_reddit_data_full_2.csv', index=False)
# with pd.ExcelWriter('turkish_reddit_data_full.xlsx', engine='xlsxwriter') as writer:
#     df.to_excel(writer, index=False)

# d_keywords = df[df['contains_keywords'] == True]
# with pd.ExcelWriter('turkish_reddit_data_controversial.xlsx', engine='xlsxwriter') as writer:
#     df_keywords.to_excel(writer, index=False)

# df_profanity = df[df['contains_profanity'] == True]
# with pd.ExcelWriter('turkish_reddit_data_profanity.xlsx', engine='xlsxwriter') as writer:
#     df_profanity.to_excel(writer, index=False)

# df_both = df[(df['contains_keywords'] == True) & (df['contains_profanity'] == True)]
# with pd.ExcelWriter('turkish_reddit_data_controversial_and_profanity.xlsx', engine='xlsxwriter') as writer:
#     df_both.to_excel(writer, index=False)

# print(f"Collection complete!")
# print(f"Collected {len(df)} total posts with {sum(df['num_comments'])} comments")
# print(f"Found {len(df_keywords)} posts containing controversial keywords")
# print(f"Found {len(df_profanity)} posts containing profanity")
# print(f"Found {len(df_both)} posts containing both controversial keywords and profanity")
import pandas as pd 
if __name__ == "__main__":
    df = pd.read_csv('turkish_reddit_data_full_2.csv')
    print(df.head())