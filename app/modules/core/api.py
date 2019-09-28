from flask import Blueprint

from .model.core import process_single

mod_core = Blueprint('core', __name__, url_prefix='/')


@mod_core.route('/infer')
def core():
    """HTTP infer endpoint
    """
    # input = "I really love the fit of this blouse, but the color is different than it seems online. The texture is excellent and it's comfortable to wear."
    # input = "Fantastic experience! If I was able to put 10 stars I would. I loved my experience it was really well done. The dogs were so cute and happy. I thought the staff were very knowledgeable they certainly know there stuff and the dogs really well. I loved how happy they were sitting on my lap. ( the dogs not the staff )    The tea and cake were delicious and made my experience perfect. What a great atmosphere and experience to have. "
    # input = "I love marvin gaye and jimmy buffet but i hate john travolta, john lennon, and john from the bible."
    input = "I've seen several movies here. Sound is really good and the theaters are clean. The staff is always happy and helpful. I think the VIP theater is the best movie experience around. I would avoid the D-box seats, the novelty wears thin in about 5 minutes then you're just left with a very uncomfortable seat compared to the regular seats."
    result = process_single(input)
    return {
        'core': result,
    }


@mod_core.route('/train')
def train():
    """HTTP train model endpoint
    """
    result = train_model()
    return result
