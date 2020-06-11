import click
import os
import pickle
import datetime
from sklearn.metrics import f1_score, accuracy_score
from ginipls.data.data_utils import load_data 
from ginipls.models.ginipls import PLS, PLS_VARIANT
from ginipls.models.hyperparameters import select_pls_hyperparameters_with_cross_val
from ginipls.config import GLOBAL_LOGGER
logger = GLOBAL_LOGGER

def init_and_train_pls(X_train, y_train, pls_type, hyerparameters_selection_nfolds, nu_range, n_components_range, only_the_first_fold):
  """"""
  best_nu, best_n_comp = select_pls_hyperparameters_with_cross_val(pls_type, X_train, y_train, nu_range, n_components_range, hyerparameters_selection_nfolds, only_the_first_fold=only_the_first_fold)
  #logger.info("selected hyperparameters : nu=%.3f, n_comp=%d" % (nu, n_comp))
  gpls = PLS(pls_type=pls_type, nu=best_nu, n_components=best_n_comp)
  gpls.fit(X_train, y_train)
  train_score = gpls.score(X_train, y_train)
  logger.info("train f1_score = %.3f" % (train_score))
  return gpls


@click.group()
def train():
    pass

@train.command("on-vectors", help="train on a csv file with vectors as rows and attributes as columns")
@click.argument('trainfilename', type=click.Path(exists=True))
@click.argument('classifierfilename', type=click.Path(), required=False)
@click.option('--label_col', type=str, default='@label', help='labels column name', show_default=True)
@click.option('--index_col', type=str, default='@id', help='texts ids column name', show_default=True)
@click.option('--col_sep', type=str, default="\t", help='column delimiter', show_default=True)
@click.option('--pls_type', type=click.Choice(PLS_VARIANT.__members__), callback=lambda c, p, v: getattr(PLS_VARIANT, v) if v else None, default=PLS_VARIANT.STANDARD, help='variant of PLS (%s)' % (' '.join('PLS_VARIANT.' + c.name for c in PLS_VARIANT)), show_default=True)
@click.option('--nu_range', type=list, default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], help='range of the hyperparameter nu of the gini_pls method', show_default=True)
@click.option('--n_components_range', type=list, default=range(1,11), help='range of the hyperparameter n_components', show_default=True)
@click.option('--hyperparams_nfolds', type=int, default=5, help='nb of folds for the selection of hyperparameters by cross-validation', show_default=True)
@click.option('--crossval_hyperparam/--no-crossval_hyperparam', default=False, help='run on all the fold or only on the first fold for the selection of hyperparameters', show_default=True)
def train_on_vectors(trainfilename, classifierfilename, label_col, index_col, col_sep, pls_type, nu_range, n_components_range, hyperparams_nfolds, crossval_hyperparam):
    """python -m ginipls train on-vectors --label_col=category --index_col=@id --crossval_hyperparam  data\processed\doris0_CHI2_ATF-train.tsv"""
    click.echo('This is the on-vectors subcommand of the train command')
    X_train, y_train, headers, ids = load_data(data=trainfilename, output_col=label_col, index_col=index_col, col_sep=col_sep)
    clf = init_and_train_pls(X_train, y_train, pls_type, hyperparams_nfolds, nu_range, n_components_range, only_the_first_fold=not crossval_hyperparam)
    if classifierfilename is None:
        # build a name for the model_file
        input_basename = os.path.basename(trainfilename).split('.')[0]
        classifierfilename = os.path.join("models","_".join([datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),input_basename, str(pls_type.name)+"PLS.model"]))
    pickle.dump(clf, open(classifierfilename, 'wb'))
    logger.info("The trained classifier is saved at %s" % (classifierfilename))

@train.command("on-texts", help="vectorize training texts then train on their vectors")
def train_on_texts():
    click.echo('This is the on-texts subcommand of the train command [UNAVAILABLE]')

#cli = click.CommandCollection(sources=[train])

@click.group()
def apply():
    pass
    
@apply.command('on-vectors', help="apply a trained pls on vectors and save outputs in outputfilename")
@click.argument('vectorsfilename', type=click.Path(exists=True))
@click.argument('classifierfilename', type=click.Path(exists=True))
@click.argument('outputfilename', type=click.Path(), required=False)
@click.option('--label_col', type=str, default=None, help='labels column name [optional]', show_default=True, required=False)
@click.option('--index_col', type=str, default='@id', help='texts ids column name', show_default=True)
@click.option('--col_sep', type=str, default="\t", help='column delimiter', show_default=True)
def apply_on_vectors(vectorsfilename, classifierfilename, outputfilename, label_col, index_col, col_sep):
    """python -m ginipls apply on-vectors --label_col=category --index_col=@id data\processed\doris0_CHI2_ATF-test.tsv models\2020-06-11_14-58-50_doris0_CHI2_ATF-train_GINIPLS.model"""
    click.echo('This is the on-vectors subcommand of the apply command')
    X_test, y_test, h, ids_test = load_data(data=vectorsfilename, output_col=label_col, index_col=index_col, col_sep=col_sep)
    clf = pickle.load(open(classifierfilename, 'rb'))
    logger.debug(clf.get_params())
    logger.info("Trained %s PLS classifier loaded from %s." % (clf.get_params()["pls_type"].name, classifierfilename))
    y_pred = clf.predict(X_test)
    if outputfilename is None:
        # build a name for the model_file
        input_basename = os.path.basename(vectorsfilename).split('.')[0]
        outputfilename = os.path.join("reports","_".join([datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),input_basename, str(pls_type.name)+"PLS.out.tsv"]))
    if label_col: # save y_pred and y_true
        outputfilename
    else:
        outputfilename

@apply.command('on-text', help="vectorize training texts then apply a trained pls on their vectors")
def on_texts():
    click.echo('This is the on-texts subcommand of the train command [UNAVAILABLE]')

@click.group()
def test():
    pass
    

@test.command('on-vectors', help="apply a trained pls on vectors")
def test_on_vectors(trainfilename, classifierfilename, label_col, index_col, col_sep):
    # test_acc_score = accuracy_score(y_test, y_pred)
    # logger.info("test accuracy_score = %.3f" % (test_acc_score))
    # test_f1_score = f1_score(y_test, y_pred, labels=[0,1], average='macro')
    # logger.info("test f1_score = %.3f" % (test_f1_score))
    pass

@click.group()
def cli():
    pass

cli.add_command(apply)
cli.add_command(test)
cli.add_command(train)

if __name__ == '__main__':
    cli()