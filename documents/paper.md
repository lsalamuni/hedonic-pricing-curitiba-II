# Pricing the Unpriced: A Hedonic Analysis of Curitiba's Segmented Real Estate Market

+++ {"part": "abstract"}

This study investigates which housing attributes most influence supplier pricing
decisions in Curitiba's residential real estate market, and how the relative importance
of these attributes varies across socioeconomic market segments. Following Rosen's
(1974) two-stage hedonic framework, I first estimate hedonic price functions to obtain
implicit marginal prices for housing area, then use these as the dependent variable in
second-stage supply equations incorporating supply shifters and locational
characteristics. Principal component analysis on six neighborhood socioeconomic
variables classifies 75 neighborhoods into three market tiers. A Wald test decisively
rejects the pooled model (F = 163.02, p < 0.001), confirming market segmentation.
Results show that the valuation of housing attributes varies substantially across tiers,
consistent with heterogeneous demand in segmented urban housing markets.

This project builds on the template by {cite}`GaudeckerEconProjectTemplates`.

+++

```{raw} latex
\clearpage
```

## Research Question

This study investigates which housing attributes most influence supplier pricing
decisions in Curitiba's residential real estate market, and how the relative importance
of these attributes varies across socioeconomic market segments. Specifically, I ask:
*Do intrinsic characteristics (area, bedrooms, bathrooms, condominium amenities) and
locational attributes (proximity to schools, hospitals, green areas, transit terminals)
exert different effects on the implicit price per square meter in low-, middle-, and
high-income neighborhoods?*

Following {cite}`Rosen74`'s two-stage hedonic framework, I first estimate a hedonic
price function to obtain implicit marginal prices for housing area, then use these as
the dependent variable in a second-stage supply equation that incorporates supply
shifters (population density, total population) and locational characteristics. This
approach, applied by {cite}`FaveroEtAl08` to Sao Paulo's metropolitan area, allows
identification of which attributes developers implicitly price into their offerings and
how these pricing strategies vary across market segments. By estimating separate
equations for each market tier, I can formally test whether coefficient heterogeneity
exists using Wald tests.

## Motivation

Understanding how housing attributes are priced across different market segments has
important implications for real estate development, urban planning, and public policy.
If the implicit prices of attributes vary systematically across socioeconomic tiers,
this suggests that Curitiba's housing market is segmented rather than integrated, a
finding with direct consequences for how developers design projects targeting specific
income groups and how policymakers evaluate the distributional effects of infrastructure
investments.

{cite}`Favero05` and {cite}`FaveroEtAl08` demonstrated this heterogeneity for Sao Paulo,
finding that attribute coefficients differ significantly across low-, middle-, and
high-income districts. {cite}`SartorisNeto96` provided earlier evidence of hedonic price
heterogeneity in the Sao Paulo market. {cite}`JohnPorsse16` applied hedonic methods to
Curitiba's apartment market but without stratifying by socioeconomic tiers. No study,
however, has combined PCA-based market segmentation with Rosen's two-stage supply-side
estimation for Curitiba, despite the city's international recognition for innovative
urban planning (particularly its Bus Rapid Transit system) and its substantial spatial
inequality. Curitiba presents a compelling case study: it combines a well-documented
planning history with marked socioeconomic stratification across its 75 neighborhoods.
This study fills this gap by applying Rosen's two-stage methodology, with a focus on the
supply side, to Curitiba's segmented housing market.

## Data and Methodology

This study employs the hedonic pricing methodology to estimate the implicit prices of
housing attributes in Curitiba's residential real estate market. Following the
theoretical framework established by {cite}`Lancaster66` and {cite}`Rosen74`, observed
housing prices are regressed on a vector of intrinsic and extrinsic characteristics,
yielding estimates of implicit marginal valuations for each attribute.

**Data.** I combine two data sources: (1) residential property listings (N = 6,833 after
outlier removal) scraped from major Brazilian real estate platforms, containing listing
prices and structural characteristics (total area, age, bedrooms, bathrooms, parking
spots, and 12 condominium amenities including pool, gym, party room, and gourmet space);
and (2) neighborhood-level data from IPPUC (Curitiba's urban planning institute),
including socioeconomic indicators (income, literacy, population density) and locational
attributes (proximity to hospitals, schools, shopping centers, green areas, bus
terminals).

**Market Segmentation.** Following {cite}`FaveroEtAl08` and the multivariate framework
in {cite}`FaveroBelfiore17`, I stratify the market into three tiers (low, middle, high)
using Principal Component Analysis on six neighborhood socioeconomic variables. The
first principal component (explaining 50.5% of variance) loads positively on income and
literacy and negatively on poverty, providing a natural socioeconomic ranking. This
stratification addresses potential identification problems that arise when estimating
hedonic models with pooled data from heterogeneous submarkets {cite}`Palmquist84`. A
Wald test formally validates that hedonic coefficients differ across segments,
justifying the stratified estimation approach.

**Empirical Strategy.** Following {cite}`Rosen74`, I employ a two-stage procedure. In
the *first stage*, I estimate a hedonic price function for each tier k:

$$
\ln(P_{ik}) = \alpha_k + \beta_k \ln(\text{AREA}_i) + \boldsymbol{\gamma}_k \mathbf{z}_{i} + \epsilon_{ik}
$$

where $\mathbf{z}_i$ is a vector of intrinsic characteristics (bedrooms, bathrooms,
parking, amenities). The log-log functional form is validated via Box-Cox tests. The
coefficient $\beta_k$ yields the implicit marginal price:
$\hat{p}_i = \beta_k \cdot P_i / \text{AREA}_i$.

In the *second stage*, I estimate the supply equation using the implicit price as the
dependent variable:

$$
\hat{p}_{ik} = f(\mathbf{z}_i, \text{AREA}_i, \mathbf{Y}_{2i}, \mathbf{W}_i)
$$

where $\mathbf{Y}_2$ contains supply shifters (population density, total population) and
$\mathbf{W}$ contains locational attributes affecting supply (proximity to schools,
hospitals, green areas, transit terminals). {cite}`Conley99` HAC standard errors address
spatial autocorrelation inherent in housing data.

All code and data files used can be found at the following GitHub repository:
[final-project-lsalamuni](https://github.com/iame-uni-bonn/final-project-lsalamuni/tree/main).

## Expected Findings

I expect the relative importance of housing attributes to vary systematically across
market tiers, reflecting different consumer preferences and supply constraints in each
segment:

- Condominium amenities (pool, gym, party room) should command higher premiums in low-
  and middle-tier markets, where they represent scarce luxury goods that signal status,
  but smaller or insignificant premiums in high-tier markets where such amenities are
  standard features expected by buyers.
- Locational attributes such as proximity to green areas, cycleways, and private schools
  should be more strongly priced in high-income neighborhoods, consistent with
  environmental quality and education access being normal goods with positive income
  elasticity.
- Parking premiums should increase monotonically with market tier, reflecting higher car
  ownership rates and land scarcity in wealthier central neighborhoods.
- Supply shifters (population density) may have opposing effects across tiers: negative
  in high-income areas (reflecting land scarcity and congestion disamenities) but
  neutral or positive in lower tiers (reflecting agglomeration benefits and
  infrastructure availability).

These findings would confirm that Curitiba's housing market is segmented rather than
integrated, implying that a single hedonic equation for the entire city would produce
biased estimates. From a practical standpoint, the results would inform developers about
which attributes to prioritize when targeting specific income groups, and provide
policymakers with evidence on how infrastructure investments are capitalized differently
across neighborhoods.

## Preliminary Results

The dataset comprises residential property listings collected from major Brazilian real
estate platforms, geocoded and merged with neighborhood-level socioeconomic indicators
from IPPUC. Following the stratification methodology proposed by {cite}`Favero05`,
Principal Component Analysis was applied to six neighborhood characteristics: average
income, proportion of households below half the minimum wage, literacy rate, population
growth rate, population density, and total population. Applying the Kaiser criterion,
the first two principal components (explaining the majority of total variance) were
retained and used to compute a weighted composite score for each neighborhood. Based on
this score, neighborhoods were classified into three tiers: low (34 neighborhoods, 2,956
observations), middle (27 neighborhoods, 2,856 observations), and high (14
neighborhoods, 1,021 observations).

To validate the market segmentation, a robust Wald test was conducted on a pooled
regression model with tier interaction terms. The test yielded an F-statistic of 163.02
with an associated p-value below $2.2 \times 10^{-16}$, strongly rejecting the null
hypothesis that hedonic coefficients are equal across tiers. This result provides robust
empirical support for estimating separate hedonic price functions for each market
segment rather than pooling observations into a single model.

Preliminary diagnostics reveal several econometric challenges. White's test indicates
the presence of heteroskedasticity in all tier models, while Moran's I test detects
significant spatial autocorrelation in the residuals across all tiers (I = 0.21 for low,
0.33 for mid, 0.23 for high; all p < 0.001). To address these issues, {cite}`Conley99`
heteroskedasticity and autocorrelation consistent standard errors are employed with a
2-kilometer distance cutoff, ensuring valid inference in the presence of spatial
dependence. Box-Cox transformations confirm that the log-log functional form provides
the best fit across all three market segments (optimal lambda values of -0.25, -0.06,
and 0.29), consistent with the hedonic pricing literature.

Initial first-stage estimates suggest that total area and property age are significant
determinants of listing prices across all tiers, though the magnitude of these
elasticities varies systematically with neighborhood socioeconomic status. Condominium
amenities such as pools, gyms, and security features appear to command higher premiums
in middle and high-tier neighborhoods. Second-stage supply estimates reveal that
locational attributes (particularly the existence of shopping centers, green areas, and
public transportation terminals) exert significant effects on the implicit price per
square meter, with coefficients varying across market tiers.

However, preliminary results also reveal several limitations that warrant attention.
Some coefficients exhibit counterintuitive negative signs, likely attributable to
multicollinearity among correlated housing attributes. Additionally, the potential for
omitted variable bias (OVB) cannot be ruled out, as unobserved neighborhood
characteristics or property features may confound the estimated relationships. These
findings suggest that a more careful selection of intrinsic and extrinsic variables
(potentially through stepwise regression or regularization techniques) should be
undertaken in the final specification.

```{raw} latex
\bibliographystyle{unsrtnat}
\bibliography{main.bib}
\renewcommand{\bibliography}[1]{}
\clearpage
```

## Appendix

### 1. PCA

```{raw} latex
\begin{table}[!ht]
\centering
\caption{PCA Variable Definitions.}\label{tab:pca-variables}
\begin{tabular}{l l}
\toprule
Variable & Description \\
\hline
Pop\_1/2\_SM & Proportion of population below half minimum wage \\
Inc & Average per-capita income (BRL) \\
Lit & Literacy rate (\%) \\
Dens & Population density (inhabitants/km2) \\
Grow & Population growth rate (\%) \\
Pop & Total neighborhood population \\
\bottomrule
\end{tabular}
\end{table}
```

```{figure} public/correlation_heatmap.png
---
width: 90%
label: fig:pca-corr
---
Correlation matrix between neighborhood socioeconomic variables.
```

```{figure} public/loadings_heatmap.png
---
width: 90%
label: fig:pca-loadings
---
Factor loadings for each variable across principal components.
```

```{figure} public/scree_plot.png
---
width: 90%
label: fig:pca-scree
---
Scree plot showing explained variance by principal component.
```

```{raw} latex
% {{ADEQUACY_TABLE}}
```

```{figure} public/pca_biplot.png
---
width: 99%
label: fig:pca-biplot
---
PCA biplot displaying variable loadings on first two principal components.
```

```{raw} latex
\clearpage
```

### 2. Rosen's Two-Stage Analysis

```{figure} public/boxcox_low.png
---
width: 45%
label: fig:boxcox-low
---
Box-Cox log-likelihood for the low tier (optimal $\lambda = -0.25$).
```

```{figure} public/boxcox_mid.png
---
width: 45%
label: fig:boxcox-mid
---
Box-Cox log-likelihood for the mid tier (optimal $\lambda = -0.06$).
```

```{figure} public/boxcox_high.png
---
width: 45%
label: fig:boxcox-high
---
Box-Cox log-likelihood for the high tier (optimal $\lambda = 0.29$).
```

```{figure} public/residuals_first_stage_low.png
---
width: 85%
label: fig:resid-first-low
---
Residual diagnostics for the first-stage low-tier regression.
```

```{figure} public/residuals_first_stage_mid.png
---
width: 85%
label: fig:resid-first-mid
---
Residual diagnostics for the first-stage mid-tier regression.
```

```{figure} public/residuals_first_stage_high.png
---
width: 85%
label: fig:resid-first-high
---
Residual diagnostics for the first-stage high-tier regression.
```

```{figure} public/residuals_second_stage_low.png
---
width: 85%
label: fig:resid-second-low
---
Residual diagnostics for the second-stage low-tier regression.
```

```{figure} public/residuals_second_stage_mid.png
---
width: 85%
label: fig:resid-second-mid
---
Residual diagnostics for the second-stage mid-tier regression.
```

```{figure} public/residuals_second_stage_high.png
---
width: 85%
label: fig:resid-second-high
---
Residual diagnostics for the second-stage high-tier regression.
```

```{raw} latex
\clearpage
\begin{table}[!ht]
\centering
\caption{Variable Definitions.}\label{tab:variables}
\scriptsize
\begin{tabular}{l l}
\toprule
Variable & Description \\
\hline
\textbf{Dependent Variables} & \\
Price & Listing price (BRL) \\
Implicit price & Marginal price per m2 \\
\textbf{Intrinsic Characteristics} & \\
Total area & Total property area (m2) \\
Age & Property age (years) \\
Off-plan & Property under construction (dummy) \\
Bedroom (1-4+) & Number of bedrooms (categorical) \\
Bathroom (1-4+) & Number of bathrooms (categorical) \\
Parking (1, 2+) & Number of parking spots (categorical) \\
Balcony & Property has balcony (dummy) \\
Party room & Building has party room (dummy) \\
Game room & Building has game room (dummy) \\
Gym & Building has gym (dummy) \\
Pool & Building has pool (dummy) \\
Sauna & Building has sauna (dummy) \\
BBQ & Building has barbecue area (dummy) \\
Gourmet space & Building has gourmet space (dummy) \\
Sports court & Building has sports court (dummy) \\
Playground & Building has playground (dummy) \\
Guardhouse & Building has guardhouse (dummy) \\
Cameras & Building has security cameras (dummy) \\
\textbf{Locational Attributes} & \\
Green area & Neighborhood has green area (dummy) \\
Cicloways & Neighborhood has cycleway (dummy) \\
Hospitals & Neighborhood has hospital (dummy) \\
Terminals & Neighborhood has bus terminal (dummy) \\
Private schools & Neighborhood has private school (dummy) \\
Public schools & Neighborhood has public school (dummy) \\
Culture facilities & Neighborhood has cultural facility (dummy) \\
Shoppings & Neighborhood has shopping center (dummy) \\
\textbf{Supply Shifters} & \\
Log(population) & Log of neighborhood population \\
Log(density) & Log of neighborhood population density \\
\bottomrule
\end{tabular}
\end{table}
```

```{raw} latex
% {{FIRST_STAGE_TABLE}}
```

```{raw} latex
% {{SECOND_STAGE_TABLE}}
```
