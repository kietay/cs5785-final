"""
Use get_tags() to add a new column to dataframe with tags as one whole phrase - e.g. sports:tennis
"""
def extract_tags_to_list(fp):
    
    with open(fp) as f:
        tags = f.readlines()

    return [tag.strip() for tag in tags]


def get_tags(df, train_or_test='train', imfile_column='image_file'):
    """Returns pandas series of image tags"""
    
    return df[imfile_column].apply(
        lambda x: extract_tags_to_list(
            f'data/tags_{train_or_test}/' + x.split('/')[-1].replace('jpg', 'txt')
        )
    )



"""
Use get_tags_split() to add two new columns to dataframe with tags split by hierarchy - e.g. [sports, tennis]
"""
def extract_tags_to_list_split(fp):
    
    with open(fp) as f:
        tags = f.readlines()
        
    tag_pairs = [tag.strip().split(':') for tag in tags]
    
    higher_cat = [x[0] for x in tag_pairs]
    lower_cat = [x[0] for x in tag_pairs]

    return higher_cat, lower_cat


def get_tags_split(df, train_or_test='train', imfile_column='image_file'):
    """Returns two pandas series of image tags, higher and lower category"""
    
    return zip(*df[imfile_column].map(
        lambda x: extract_tags_to_list_split(
            f'data/tags_{train_or_test}/' + x.split('/')[-1].replace('jpg', 'txt')
        )
    ))
