import praw
import pandas as pd
import datetime
import re
import emoji

# Authentication setup
with open('../credentials.txt', 'r') as file:
    secret, client_id = file.read().split(';')
user_agent = "windows:tr.DILBERT:v1.0 (by /u/imemmul)"

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=secret,
    user_agent=user_agent,
)

# Target subreddits
subreddits = ['Turkey', 'turkish', 'TurkeyJerky']

# Define keyword lists
controversial_keywords = [
    'mülteci', 'suriyeli', 'göçmen', 'ermeni', 'kürt', 
    'yunan', 'azınlık', 'siyaset', 'iktidar', 'muhalefet', 'tayyip',
    'rum', 'yahudi', 'şeriat', 'şeriatçı', 'müslüman', 'ateist', 'alevi',
    'arap', 'karaboğa', 'suriye', 'fetö', 'hdp', 'pkk', 'lgbt', 'eşcinsel',
    'trans', 'kadın', 'tecavüz', 'taciz', 'ahlak', 'süryani', 'ırkçı', 'ırkçılık'
]

hate_keywords = [
    'siktir', 'amcık', 'am', 'göt', 'orospu', 'piç', 'yavşak', 'ibne', 
    'pezevenk', 'yarrak', 'çük', 'bok', 'gerizekalı', 'salak', 'aptal',
    'mal', 'sikik', 'gavat', 'amk', 'aq', 'sg', 'siktir git', 'mk',
    'yarak', 'dalyarak', 'oç', 'puşt', 'sürtük', 'kahpe', 'fahişe', 'haysiyetsiz',
    'defol', 'serefsiz', 'şerefsiz', 'haysiyetsiz', 'aşağılık'
]

emotional_indicators = ['!', '?', '!!!', '???', '!?', 'ÖSĞİ', 'ÖÇŞĞÜ']

# Track unique text entries to avoid duplicates
all_texts = set()
annotation_data = []

# Target number of entries for the dataset
TARGET_COUNT = 2000

# Number of comments to fetch per post
COMMENTS_LIMIT = 200  # Increased to get more comments to filter

# Minimum word count to consider
MIN_WORD_COUNT = 5

# Track progress
post_count = 0
comment_count = 0

print(f"Starting focused data collection to create a hate speech annotation dataset...")

def contains_controversial_keywords(text):
    """Check if text contains any controversial keywords."""
    if not text:
        return False
    text = text.lower()
    for keyword in controversial_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            return True
    return False

def contains_hate_keywords(text):
    """Check if text contains any hate speech keywords."""
    if not text:
        return False
    text = text.lower()
    for keyword in hate_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            return True
    return False

def contains_emotional_language(text):
    """Check if text contains emotional indicators like exclamation marks or ALL CAPS."""
    if not text:
        return False
    
    # Check for exclamation marks, question marks, etc.
    for indicator in emotional_indicators:
        if indicator in text:
            return True
    
    # Check for ALL CAPS words (at least 4 characters)
    words = text.split()
    for word in words:
        if len(word) >= 4 and word.isupper():
            return True
    
    return False

def is_too_short(text):
    """Check if text is too short (less than MIN_WORD_COUNT words)."""
    if not text:
        return True
    
    # Remove emojis for word counting
    text_no_emoji = emoji.replace_emoji(text, replace='')
    
    # Count words
    words = [w for w in text_no_emoji.split() if len(w) > 1]
    return len(words) < MIN_WORD_COUNT

def is_likely_hate_speech(text, score=None):
    """Calculate a hate speech likelihood score for the text."""
    if not text or is_too_short(text):
        return False, 0
    
    score = 0
    
    # Check for controversial topics
    if contains_controversial_keywords(text):
        score += 3
    
    # Check for hate keywords
    if contains_hate_keywords(text):
        score += 5
    
    # Check for emotional language
    if contains_emotional_language(text):
        score += 2
    
    # Low score comments (less than 3) likely don't contain hate speech
    return score >= 3, score

# Calculate the percentage of posts to collect from each sort method
posts_to_collect = {
    'controversial': 0.4,  # 40% from controversial 
    'top': 0.1,            # 10% from top
    'hot': 0.2,            # 20% from hot
    'new': 0.3             # 30% from new
}

# Collect data from specified subreddits
for subreddit_name in subreddits:
    print(f"\nProcessing subreddit: {subreddit_name}")
    subreddit = reddit.subreddit(subreddit_name)
    
    # Try different sorting methods to get varied content
    for sort_method in ['controversial', 'hot', 'new', 'top']:
        # Calculate target count for this sort method
        method_target = int(TARGET_COUNT * posts_to_collect[sort_method])
        method_count = 0
        
        if len(annotation_data) >= TARGET_COUNT:
            break
            
        print(f"  Using sort method: {sort_method} (target: {method_target})")
        
        # Apply time filter for certain sort methods
        if sort_method in ['top', 'controversial']:
            posts_method = getattr(subreddit, sort_method)(limit=150, time_filter='all')
        else:
            posts_method = getattr(subreddit, sort_method)(limit=150)

        # Process each post
        for submission in posts_method:
            post_count += 1
            
            # Check if the post title might contain hate speech
            if submission.title and submission.title.strip() and submission.title not in all_texts:
                is_hate, score = is_likely_hate_speech(submission.title)
                if is_hate:
                    all_texts.add(submission.title)
                    annotation_data.append({
                        'text': submission.title, 
                        'label': '', 
                        'source': 'post_title',
                        'score': submission.score,
                        'hate_likelihood': score
                    })
                    method_count += 1
            
            # Check if the post body might contain hate speech
            if submission.selftext and submission.selftext.strip() and submission.selftext not in all_texts:
                is_hate, score = is_likely_hate_speech(submission.selftext)
                if is_hate:
                    all_texts.add(submission.selftext)
                    annotation_data.append({
                        'text': submission.selftext, 
                        'label': '', 
                        'source': 'post_body',
                        'score': submission.score,
                        'hate_likelihood': score
                    })
                    method_count += 1
            
            # Process comments for this submission, prioritizing controversial ones
            try:
                submission.comment_sort = 'controversial'
                submission.comments.replace_more(limit=10)  # Replace more MoreComments objects
            except:
                pass  # If comment sorting fails, continue with default
            
            # Process each comment
            for comment in submission.comments.list()[:COMMENTS_LIMIT]:
                comment_count += 1
                
                # Skip if comment body is empty or already in dataset
                if not comment.body or not comment.body.strip() or comment.body in all_texts:
                    continue
                
                # Check if the comment might contain hate speech
                is_hate, score = is_likely_hate_speech(comment.body)
                if is_hate:
                    all_texts.add(comment.body)
                    annotation_data.append({
                        'text': comment.body, 
                        'label': '', 
                        'source': 'comment',
                        'score': comment.score,
                        'controversiality': getattr(comment, 'controversiality', 0),
                        'hate_likelihood': score
                    })
                    method_count += 1
                
                # Check if we've reached our target for this method
                if method_count >= method_target:
                    break
            
            # Check if we've reached our target for this method
            if method_count >= method_target:
                break
                
            # Progress update
            if post_count % 10 == 0:
                print(f"Processed {post_count} posts, {comment_count} comments")
                print(f"Collected {len(annotation_data)} potential hate speech texts ({method_count} from {sort_method})")
        
        print(f"Completed {sort_method} collection: {method_count}/{method_target} texts")
        
        # Check if we've reached our overall target
        if len(annotation_data) >= TARGET_COUNT:
            print(f"Reached target of {TARGET_COUNT} texts, stopping collection")
            break

# Sort by hate speech likelihood score
annotation_df = pd.DataFrame(annotation_data)
annotation_df = annotation_df.sort_values('hate_likelihood', ascending=False)

# Trim to exactly TARGET_COUNT entries if we have more
if len(annotation_df) > TARGET_COUNT:
    annotation_df = annotation_df.head(TARGET_COUNT)

# Remove all metadata columns before saving (we don't want to bias the annotation)
final_df = annotation_df[['text', 'label']]

# Shuffle the data to avoid having all high-score texts first
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Save the dataset
final_df.to_csv('turkish_reddit_hate_speech_dataset.csv', index=False)

# Save a backup with the full information for analysis
annotation_df.to_csv('turkish_reddit_hate_speech_dataset_with_metadata.csv', index=False)

print(f"\nCollection complete!")
print(f"Processed {post_count} posts and {comment_count} comments")
print(f"Created focused hate speech dataset with {len(final_df)} entries saved to 'turkish_reddit_hate_speech_dataset.csv'")
print(f"A backup with additional metadata was saved to 'turkish_reddit_hate_speech_dataset_with_metadata.csv'")

# Print statistics
print("\nData source breakdown:")
source_counts = final_df['source'].value_counts()
for source, count in source_counts.items():
    print(f"- {source}: {count} texts ({count/len(final_df)*100:.1f}%)")