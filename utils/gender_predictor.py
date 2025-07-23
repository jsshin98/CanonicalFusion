from utils.interrogator import Interrogator, WaifuDiffusionInterrogator

interrogator = WaifuDiffusionInterrogator(
    'wd14-convnextv2-v2',
    repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    revision='v2.0'
)


def predict_gender(image):
    """
    Make predictions from image paths
    """
    result = interrogator.interrogate(image)
    init_tags = Interrogator.postprocess_tags(result[1], threshold=0.7)
    tags_str = ", ".join(init_tags.keys())
    if 'boy' in tags_str:
        return 'male'
    elif 'girl' in tags_str:
        return 'female'
    else:
        return 'neutral'
