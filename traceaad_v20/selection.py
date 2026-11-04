"""Cold-start relative-gap P and log-logistic h, with independent Laplace models.

P uses log1p(gap/SCALE) features, so a feature is zero at a zero gap and grows
smoothly. Sampled P coefficients obey the signs the mechanism implies: a worse
parent or more failures never raise P(beat T); more successes never lower it.
"""
import math
import numpy as np

SCALE = .01
CENTER = math.log(SCALE)
H_MAX_SLOPE = -.05
FEATURES = ['intercept', 'g_positive', 'log1p_g_scaled', 'v_positive',
            'log1p_v_scaled', 'log1p_k_same', 'log1p_k_other', 'log1p_s']
SIGNS = {'v_positive': -1, 'log1p_v_scaled': -1, 'log1p_k_same': -1,
         'log1p_k_other': -1, 'log1p_s': 1}


def feature_names(value):
    return FEATURES[:1] + (['crossover'] if value == 'X' else []) + FEATURES[1:]


def constrain(beta, value):
    """Clip coefficients to the side their mechanism allows."""
    beta = np.array(beta, float)
    for i, name in enumerate(feature_names(value)):
        sign = SIGNS.get(name)
        if sign is not None and sign * beta[i] < 0:
            beta[i] = 0.
    return beta


def principle_prior():
    values, h_values = {}, {}
    for value in ('R', 'X'):
        mean = [math.log(.1/.9), 0., 0., 0., 0., 0., 0., 0.]
        variance = [2.25, 1., 1., 1., 1., 1., 1., 1.]
        if value == 'X':
            mean.insert(1, 0.); variance.insert(1, 1.)
        values[value] = {'mean': mean, 'precision': np.diag(1/np.array(variance)).tolist(),
                         'covariance': np.diag(variance).tolist()}
        h_values[value] = {'mean': [0., -1.], 'precision': np.diag([1/2.25, 1.]).tolist(),
                           'covariance': np.diag([2.25, 1.]).tolist()}
    return {'values': values, 'h_values': h_values}


def sigmoid(z):
    z = np.asarray(z)
    return np.exp(-np.logaddexp(0., -z))


def laplace(x, y, mean, precision, start=None):
    """Newton MAP with line search; Fisher curvature supplies the covariance."""
    mean, precision = np.asarray(mean, float), np.asarray(precision, float)
    x = np.asarray(x, float).reshape(-1, len(mean))
    y = np.asarray(y, float)
    beta = mean.copy() if start is None else np.asarray(start, float).copy()
    def objective(b):
        d = b - mean
        return float(np.logaddexp(0, x @ b).sum() - y @ (x @ b) + .5 * d @ precision @ d)
    for iteration in range(100):
        prob = sigmoid(x @ beta)
        gradient = x.T @ (prob - y) + precision @ (beta - mean)
        hessian = (x.T * (prob * (1 - prob))) @ x + precision
        step = np.linalg.solve(hessian, gradient)
        if np.max(np.abs(step)) < 1e-9 or np.max(np.abs(gradient)) < 1e-8:
            break
        rate, old = 1., objective(beta)
        while rate > 1e-10 and objective(beta - rate * step) > old - 1e-4 * rate * float(gradient @ step) + 1e-12 * (1 + abs(old)):
            rate *= .5
        beta -= rate * step
    else:
        raise RuntimeError('Laplace Newton iteration did not converge')
    prob = sigmoid(x @ beta)
    information = (x.T * (prob * (1 - prob))) @ x + precision
    return beta, np.linalg.inv(information), {'iterations': iteration + 1,
        'gradient_max': float(np.max(np.abs(x.T @ (prob-y) + precision @ (beta-mean))))}


def reference_pool(archive, parent_id):
    candidates = sorted((p for p in archive.values() if p.valid),
                        key=lambda p: (p.evaluation.fitness, p.id))
    if not candidates:
        return []
    n = len(candidates)
    median = candidates[n // 2].evaluation.fitness
    if n % 2 == 0:
        median = median / 2 + candidates[n // 2 - 1].evaluation.fitness / 2
    return [p for p in candidates if p.id != parent_id and p.evaluation.fitness <= median]


def path_best(archive, pid, depth=8):
    p = archive[pid]
    values = [p.evaluation.fitness] if p.valid else []
    seen = {pid}
    for _ in range(depth):
        pid = p.parent_id
        if pid in (None, 0) or pid in seen or pid not in archive:
            break
        seen.add(pid)
        p = archive[pid]
        if p.valid:
            values.append(p.evaluation.fitness)
    return min(values)


def design_matrix(fitness, thresholds, best, counts, value, crossover=False):
    """Rows are decision-time features. Counts columns are R, X and successes."""
    f, t, counts = np.asarray(fitness), np.asarray(thresholds), np.asarray(counts)
    scale = max(abs(best), 1e-12)
    g, v = np.maximum(0., (t-best)/scale), np.maximum(0., (f-t)/scale)
    gp, vp = g>0, v>0
    lg, lv = np.log1p(g/SCALE), np.log1p(v/SCALE)
    ix = 0 if value == 'R' else 1
    columns = [np.ones(len(f))]
    if value == 'X': columns.append(np.full(len(f), float(crossover)))
    columns.extend([gp, lg, vp, lv, np.log1p(counts[:,ix]),
                    np.log1p(counts[:,1-ix]), np.log1p(counts[:,2])])
    return np.column_stack(columns), g, v


def features(fitness, threshold, best, counts, value, crossover=False):
    counts = [[counts.get('R',0), counts.get('X',0), counts.get('s',0)]]
    x, g, v = design_matrix([fitness], [threshold], best, counts, value, crossover)
    return x[0].tolist(), float(g[0]), float(v[0])


def conversion(g, parameters):
    if g == 0: return 1.
    return float(sigmoid(parameters[0] + parameters[1]*(math.log(g)-CENTER)))


def gaussian_sample(model, rng):
    mean = np.asarray(model['mean'])
    return mean + np.linalg.cholesky(np.asarray(model['covariance'])) @ np.array([rng.gauss(0,1) for _ in mean])


def frozen_decision(ids, fitness, thresholds, counts, data, rng, *,
                    weights=(.45,.30,.25), sample=True, distinct_crossover=True):
    """Vectorized decision shared by search and offline replay."""
    f = np.asarray(fitness)
    median = float(np.median(f)); pool = {i for i,a in zip(ids,f) if a <= median}
    cross_ok = np.array([bool(pool-{pid}) for pid in ids])
    op = rng.choices(['Refine','Explore','Crossover'],
        [weights[0],weights[1],weights[2] if np.any(cross_ok) else 0.],k=1)[0]
    requested_op=op
    if op=='Crossover' and not distinct_crossover:op='Explore'
    value = 'R' if op == 'Refine' else 'X'
    model, hm = data['models'][value], data['h_models'][value]
    raw_beta = gaussian_sample(model,rng) if sample else np.asarray(model['mean'],float)
    beta = constrain(raw_beta,value)
    raw_eta = gaussian_sample(hm,rng) if sample else np.asarray(hm['mean'])
    eta = raw_eta.copy()
    if sample: eta[1] = min(eta[1], H_MAX_SLOPE)
    best = float(f.min())
    x,g,v = design_matrix(f,thresholds,best,counts,value,op=='Crossover')
    probability = sigmoid(x @ beta); h = np.ones(len(ids)); mask = g>0
    h[mask] = sigmoid(eta[0]+eta[1]*(np.log(g[mask])-CENTER))
    score = probability*h
    eligible = cross_ok if op=='Crossover' else np.ones(len(ids),dtype=bool)
    maximum = score[eligible].max()
    chosen = rng.choice(np.flatnonzero((score==maximum)&eligible).tolist())
    return dict(operator=op,requested_operator=requested_op,value=value,beta=beta,h_parameters=eta,
        best=best,x=x,g=g,v=v,probability=probability,h=h,S=score,eligible=eligible,chosen=chosen)


class Selector:
    """Posterior models for P and h; the counts record each parent's outcomes."""

    def __init__(self):
        prior = principle_prior()
        self.data = {'prior': prior, 'counts': {}, 'models': {}, 'h_models': {}}
        for value in ('R', 'X'):
            for key, prior_key in [('models', 'values'), ('h_models', 'h_values')]:
                p = prior[prior_key][value]
                self.data[key][value] = {'mean': p['mean'], 'covariance': p['covariance'], 'x': [], 'y': []}

    def choose(self, archive, config, rng, *, distinct_crossover=True):
        """Return parent, operator, reference, the decision record and all candidate rows."""
        parents = sorted((p for p in archive.values() if p.valid), key=lambda p: p.id)
        ids = [p.id for p in parents]
        fitness = [p.evaluation.fitness for p in parents]
        thresholds = [path_best(archive, p.id, config.ancestor_count) for p in parents]
        counts = [[self.data['counts'].get(str(pid), {}).get(k, 0) for k in ('R', 'X', 's')] for pid in ids]
        result = frozen_decision(ids, fitness, thresholds, counts, self.data, rng,
            weights=(config.refine_weight, config.explore_weight, config.crossover_weight),
            distinct_crossover=distinct_crossover)
        rows = [{'id': pid, 'features': result['x'][i].tolist(), 'T': thresholds[i],
                 **{k: float(result[k][i]) for k in ('g', 'v', 'probability', 'h', 'S')},
                 'eligible': bool(result['eligible'][i])} for i, pid in enumerate(ids)]
        selected = rows[result['chosen']]
        parent = archive[selected['id']]
        op, value = result['operator'], result['value']
        reference = rng.choice(reference_pool(archive, parent.id)) if op == 'Crossover' else None
        decision = {'operator': op, 'requested_operator': result['requested_operator'], 'value': value,
            'beta': result['beta'].tolist(), 'h_parameters': result['h_parameters'].tolist(),
            'posterior_mean': self.data['models'][value]['mean'],
            'h_posterior_mean': self.data['h_models'][value]['mean'],
            'best': result['best'], 'selected': selected}
        return parent, op, reference, decision, rows

    def learn(self, value, x, g, exceeded, breakthrough):
        """Use one observed actual action. Frozen x and g precede its outcome."""
        model = self.data['models'][value]
        model['x'].append(list(x)); model['y'].append(int(exceeded))
        prior = self.data['prior']['values'][value]
        mean, cov, _ = laplace(model['x'], model['y'], prior['mean'], prior['precision'], model['mean'])
        model.update(mean=mean.tolist(), covariance=cov.tolist())
        if exceeded and g > 0:
            hm = self.data['h_models'][value]
            hm['x'].append([1., math.log(g) - CENTER]); hm['y'].append(int(breakthrough))
            hp = self.data['prior']['h_values'][value]
            mean, cov, _ = laplace(hm['x'], hm['y'], hp['mean'], hp['precision'], hm['mean'])
            hm.update(mean=mean.tolist(), covariance=cov.tolist())

    def observe(self, record, parent_id, child):
        """Learn from one selection record and its final child; also used to rebuild on resume."""
        selected = record['selected']
        valid = bool(child and child.valid)
        exceeded = valid and child.evaluation.fitness < selected['T']
        breakthrough = valid and child.evaluation.fitness < record['best']
        self.learn(record['value'], selected['features'], selected['g'], exceeded, breakthrough)
        counts = self.data['counts'].setdefault(str(parent_id), {'R': 0, 'X': 0, 's': 0})
        counts['s' if exceeded else record['value']] += 1
        record['outcome'] = {'valid': valid, 'exceeded': exceeded, 'breakthrough': breakthrough,
            'fitness': child.evaluation.fitness if valid else None}
