# Apuntes

## 1. Estimación por Máxima Verosimilitud (MLE) bajo Distribución de Poisson

Supongamos que las observaciones $\{(x_i, y_i)\}_{i=1}^{n}$ son estadísticamente independientes y siguen una distribución de Poisson. Bajo la asunción *Naive Bayes*, la función de verosimilitud es el producto de las probabilidades individuales:

$$L(\theta) = \prod_{i=1}^{n} p(y_i \mid x_i, \theta) = \prod_{i=1}^{n} \frac{h_\theta(x_i)^{y_i} e^{-h_\theta(x_i)}}{y_i!}$$

donde el modelo lineal para la tasa esperada está dado por $h_\theta(x_i) = \theta_0 + \theta_1 x_i > 0$. 

Aplicando el logaritmo natural, la log-verosimilitud convierte la productoria en una sumatoria:

$$\begin{aligned}
\ell(\theta) = \ln [L(\theta)] &= \ln \left[ \prod_{i=1}^{n} \frac{h_\theta(x_i)^{y_i} e^{-h_\theta(x_i)}}{y_i!} \right] \\
&= \sum_{i=1}^{n} \left[ y_i \ln h_\theta(x_i) - h_\theta(x_i) - \ln(y_i!) \right]
\end{aligned}$$

De acuerdo con el criterio de Estimación por Máxima Verosimilitud (EMV / MLE), maximizar $\ell(\theta)$ equivale a minimizar $-\ell(\theta)$. Además, se pueden ignorar los términos independientes de $\theta$, dado que $\frac{\partial [\ln(y_i!)]}{\partial \theta} = 0$:

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta \in \Theta} [\ell(\theta)] \iff \arg\min_{\theta \in \Theta} [-\ell(\theta)]$$

Se define entonces la función de costo $J(\theta)$ como:

$$J(\theta) := \sum_{i=1}^{n} \left[ h_\theta(x_i) - y_i \ln h_\theta(x_i) \right]$$

### Cálculo del Gradiente e Interpretación Física

El gradiente de la función de costo respecto a un parámetro $\theta_k$ es:

$$\frac{\partial J}{\partial \theta_k} = \sum_{i=1}^{n} \frac{\partial}{\partial \theta_k} \left[ h_\theta(x_i) - y_i \ln h_\theta(x_i) \right] = \sum_{i=1}^{n} \left[ 1 - \frac{y_i}{h_\theta(x_i)} \right] \frac{\partial h_\theta(x_i)}{\partial \theta_k}$$

Para el modelo lineal $h_\theta(x_i) = \theta_0 + \theta_1 x_i$, las derivadas parciales son:

$$\frac{\partial h_\theta(x_i)}{\partial \theta_k} = \frac{\partial}{\partial \theta_k} [\theta_0 + \theta_1 x_i] \implies \begin{cases} \frac{\partial h_\theta(x_i)}{\partial \theta_0} = 1 \\ \frac{\partial h_\theta(x_i)}{\partial \theta_1} = x_i \end{cases}$$

**Interpretación de los parámetros:**
* **$h_\theta(x_i)$**: Representa la tasa de cuentas esperada según el modelo, verificándose que $h_\theta(0) = \theta_0$.
* **Tasa de fondo ($\theta_0$)**: Si la condición experimental $x_i$ es nula, sin una tasa de fondo $\theta_0$, el modelo no sería adecuado ya que la predicción también se anularía ($\theta_0 = 0 \implies h_\theta(x_i = 0) = 0$).
* **Proporcionalidad ($\theta_1$)**: Representa una constante de proporcionalidad de primer orden asociada al comportamiento lineal de la condición experimental $x_i$. Definir $h_\theta(x_i) = \theta_0 + \theta_1 x_i$ describe un hiperplano de la forma más general.

Sustituyendo el modelo lineal, las componentes explícitas del gradiente resultan en:

$$\frac{\partial J}{\partial \theta_0} = \sum_{i=1}^{n} \left[ \left(\frac{\theta_0 + \theta_1 x_i}{\theta_0 + \theta_1 x_i}\right) - \left(\frac{y_i}{\theta_0 + \theta_1 x_i}\right) \right] = \sum_{i=1}^{n} \left[ \frac{\theta_0 + \theta_1 x_i - y_i}{\theta_0 + \theta_1 x_i} \right]$$

$$\frac{\partial J}{\partial \theta_1} = \sum_{i=1}^{n} x_i \left[ \frac{\theta_0 + \theta_1 x_i - y_i}{\theta_0 + \theta_1 x_i} \right]$$

---

## 2. Retropropagación (Backpropagation) con Función de Pérdida de Poisson

Partiendo de la distribución de probabilidad condicional de Poisson y su log-verosimilitud negativa para la observación $i$-ésima:

$$P(Y = y_i \mid x_i) = \frac{h_\theta(x_i)^{y_i} \exp(-h_\theta(x_i))}{y_i!}$$

$$-\ln P(Y = y_i \mid x_i) = -y_i \ln h_\theta(x_i) + h_\theta(x_i) + \underbrace{\ln(y_i!)}_{\nabla_\theta \to 0}$$

Definimos la pérdida individual $J_i(\theta)$ parametrizando la salida de la red mediante la función de enlace canónica exponencial $h_\theta(x_i) = \exp\left(z_{i,1}^{[2]}\right)$:

$$J_i(\theta) = \underbrace{h_\theta(x_i)}_{\exp\left(z_{i,1}^{[2]}\right)} - y_i \underbrace{\ln h_\theta(x_i)}_{z_{i,1}^{[2]}} = \exp\left(z_{i,1}^{[2]}\right) - y_i z_{i,1}^{[2]}$$

donde el parámetro natural viene dado por:

$$\eta_{i,1} = z_{i,1}^{[2]} = \ln h_\theta(x_i)$$

*(Nota: En modelos de clasificación binaria con probabilidad $p_i \equiv P(Y = 1 \mid x_i)$, el parámetro natural análogo se conoce como log-momio o logit: $\ln\left(\frac{p_i}{1 - p_i}\right)$).*

### Propagación del Error y Cálculo de Gradientes

* **Error en la capa de salida ($\delta_{i,1}^{[2]}$):**

$$\delta_{i,1}^{[2]} = \frac{\partial J_i}{\partial \eta_{i,1}} = \frac{\partial}{\partial \eta_{i,1}} \left[ e^{\eta_{i,1}} - y_i \eta_{i,1} \right] = e^{\eta_{i,1}} - y_i = h_\theta(x_i) - y_i$$

* **Gradiente en la capa de salida:**
  Dadas las pre-activaciones $z_{i,1}^{[2]} = \theta_{1,1}^{[2]} A_{i,1}^{[1]} + b_1^{[2]}$ y $z_{i,1}^{[1]} = \theta_{1,1}^{[1]} x_i + b_1^{[1]}$, con activación oculta $A_{i,1}^{[1]} = \sigma\left(z_{i,1}^{[1]}\right)$:

$$\frac{\partial J_i}{\partial \theta_{1,1}^{[2]}} = \frac{\partial J_i}{\partial z_{i,1}^{[2]}} \cdot \frac{\partial z_{i,1}^{[2]}}{\partial \theta_{1,1}^{[2]}} = \delta_{i,1}^{[2]} A_{i,1}^{[1]} = \left[ h_\theta(x_i) - y_i \right] \sigma\left(z_{i,1}^{[1]}\right)$$

$$\frac{\partial J_i}{\partial b_1^{[2]}} = \frac{\partial J_i}{\partial z_{i,1}^{[2]}} \cdot \underbrace{\frac{\partial z_{i,1}^{[2]}}{\partial b_1^{[2]}}}_{=1} = \delta_{i,1}^{[2]} = h_\theta(x_i) - y_i$$

* **Backward pass (Error propagado hacia la capa oculta):**
  Aplicando la regla de la cadena:

$$\delta_{i,1}^{[1]} = \frac{\partial J_i}{\partial z_{i,1}^{[1]}} = \underbrace{\left[ \frac{\partial J_i}{\partial z_{i,1}^{[2]}} \right]}_{\delta_{i,1}^{[2]}} \underbrace{\left[ \frac{\partial z_{i,1}^{[2]}}{\partial A_{i,1}^{[1]}} \right]}_{\theta_{1,1}^{[2]}} \underbrace{\left[ \frac{\partial A_{i,1}^{[1]}}{\partial z_{i,1}^{[1]}} \right]}_{\sigma'\left(z_{i,1}^{[1]}\right)}$$

$$\delta_{i,1}^{[1]} = \delta_{i,1}^{[2]} \theta_{1,1}^{[2]} \sigma'\left(z_{i,1}^{[1]}\right) = \left[ h_\theta(x_i) - y_i \right] \theta_{1,1}^{[2]} \sigma'\left(z_{i,1}^{[1]}\right)$$

* **Gradiente en la capa de entrada (oculta):**
  A partir de $z_{i,1}^{[1]} = \theta_{1,1}^{[1]} x_i + b_1^{[1]}$:

$$\frac{\partial J_i}{\partial \theta_{1,1}^{[1]}} = \left[ \frac{\partial J_i}{\partial z_{i,1}^{[1]}} \right] \left[ \frac{\partial z_{i,1}^{[1]}}{\partial \theta_{1,1}^{[1]}} \right] = \delta_{i,1}^{[1]} x_i = \left[ h_\theta(x_i) - y_i \right] \theta_{1,1}^{[2]} \sigma'\left(z_{i,1}^{[1]}\right) x_i$$

$$\frac{\partial J_i}{\partial b_1^{[1]}} = \left[ \frac{\partial J_i}{\partial z_{i,1}^{[1]}} \right] \underbrace{\left[ \frac{\partial z_{i,1}^{[1]}}{\partial b_1^{[1]}} \right]}_{=1} = \delta_{i,1}^{[1]} = \left[ h_\theta(x_i) - y_i \right] \theta_{1,1}^{[2]} \sigma'\left(z_{i,1}^{[1]}\right)$$

---

## 3. Redes Neuronales Informadas por la Física (PINN): Ecuación de Onda en Membrana Cuadrada

Consideramos la ecuación de onda bidimensional:

$$\frac{\partial^2 U}{\partial t^2}(x, y, t) = \frac{\partial^2 U}{\partial x^2} + \frac{\partial^2 U}{\partial y^2}$$

Sujeta a las condiciones de una membrana cuadrada unitaria mediante el *ansatz* de separación de variables:

$$\begin{cases}
U = 0 & \text{nulo en los bordes } \forall x, y \in \{0, 1\} \\
U(x, y, 0) = \sin(\pi x)\sin(\pi y) & \text{condición inicial de posición} \\
U_t(x, y, 0) = \sqrt{2}\pi U(x, y, 0) & \text{condición inicial de velocidad} \\
U_\theta(x, y, t) = U(x, y, 0) a_\theta(t) & \text{solución propuesta}
\end{cases}$$

Por construcción del *ansatz*, se satisface exactamente la condición de frontera, de modo que $[U_\theta = 0, \ J_{\text{frontera}} = 0] \ \forall x, y \in \{0, 1\}$.

Se estudian dos arquitecturas neuronales para la función temporal $a_\theta(t)$:
1. **Modelo A (una capa con dos neuronas):** $a_\theta(t) = \sum_{j=1}^{2} A_j \sin(\omega_j t + b_j)$
2. **Modelo B (dos capas con una neurona por capa):** $a_\theta(t) = A \sin[\omega_2 h_1(t) + b_2]$, donde $h_1(t) = \sin(\omega_1 t + b_1)$

### Derivadas Parciales y Reducción al Oscilador Armónico

Calculando las derivadas parciales de $U_\theta(x, y, t) = \sin(\pi x)\sin(\pi y)a_\theta(t)$:

$$\frac{\partial U_\theta}{\partial t} = \sin(\pi x)\sin(\pi y)\dot{a}_\theta(t), \quad \frac{\partial^2 U_\theta}{\partial t^2} = \sin(\pi x)\sin(\pi y)\ddot{a}_\theta(t)$$

$$\frac{\partial U_\theta}{\partial x} = \pi \cos(\pi x)\sin(\pi y)a_\theta(t), \quad \frac{\partial^2 U_\theta}{\partial x^2} = -\pi^2 \sin(\pi x)\sin(\pi y)a_\theta(t)$$

$$\frac{\partial U_\theta}{\partial y} = \pi \sin(\pi x)\cos(\pi y)a_\theta(t), \quad \frac{\partial^2 U_\theta}{\partial y^2} = -\pi^2 \sin(\pi x)\sin(\pi y)a_\theta(t)$$

Sustituyendo en la ecuación de onda:

$$\sin(\pi x)\sin(\pi y)\ddot{a}_\theta(t) = -2\pi^2 \sin(\pi x)\sin(\pi y)a_\theta(t)$$

$$[\sin(\pi x)\sin(\pi y)] \left[ \ddot{a}_\theta(t) + 2\pi^2 a_\theta(t) \right] = 0$$

Para un oscilador armónico simple se cumple:

$$\ddot{a}_\theta(t) + \omega_0^2 a_\theta(t) = 0 \implies \omega_0^2 \equiv 2\pi^2 \implies \omega_0 = \sqrt{2}\pi$$

Evaluando las condiciones iniciales en $t = 0$:

$$U_\theta(x, y, 0) = \sin(\pi x)\sin(\pi y)a_\theta(0) = \sin(\pi x)\sin(\pi y) \implies a_\theta(0) = 1$$

$$\frac{\partial U_\theta}{\partial t}(x, y, 0) = \sin(\pi x)\sin(\pi y)\dot{a}_\theta(0) = \underbrace{\sqrt{2}\pi \sin(\pi x)\sin(\pi y)}_{U_t = \sqrt{2}\pi U(x, y, 0)} \implies \dot{a}_\theta(0) = \sqrt{2}\pi$$

### Conservación de la Energía Mecánica

La energía mecánica total del sistema se conserva ($\frac{dE_\theta(t)}{dt} = 0$):

$$E_\theta(t) = \frac{1}{2} \int_0^1 \int_0^1 \left[ \left(\frac{\partial U_\theta}{\partial x}\right)^2 + \left(\frac{\partial U_\theta}{\partial y}\right)^2 + \left(\frac{\partial U_\theta}{\partial t}\right)^2 \right] dx \, dy$$

Utilizando las propiedades de ortonormalidad en el intervalo $[0, 1]$:

$$\int_0^1 \sin^2(\pi z) \, dz = \frac{1}{2}, \quad \int_0^1 \cos^2(\pi z) \, dz = \frac{1}{2}$$

Integrando cada término espacialmente (para $z \in \{x, y\}$):

$$\int_0^1 \int_0^1 \left(\frac{\partial U_\theta}{\partial t}\right)^2 dx \, dy = \int_0^1 \int_0^1 \left(\sin(\pi x)\sin(\pi y)\dot{a}_\theta(t)\right)^2 dx \, dy = \frac{1}{2} \cdot \frac{1}{2} \cdot \dot{a}_\theta^2(t) = \frac{1}{4}\dot{a}_\theta^2(t)$$

$$\int_0^1 \int_0^1 \left(\frac{\partial U_\theta}{\partial z}\right)^2 dx \, dy = \frac{1}{2} \cdot \frac{1}{2} \cdot \pi^2 a_\theta^2(t) = \frac{\pi^2}{4}a_\theta^2(t)$$

Por lo tanto, la energía en función del tiempo queda expresada como:

$$E_\theta(t) = \frac{1}{2} \cdot \frac{1}{4} \left[ \dot{a}_\theta^2(t) + \pi^2 a_\theta^2(t) + \pi^2 a_\theta^2(t) \right] = \frac{1}{8} \left[ \dot{a}_\theta^2(t) + 2\pi^2 a_\theta^2(t) \right]$$

Verificando su derivada temporal:

$$\frac{dE_\theta}{dt} = \frac{1}{8} \frac{d}{dt} \left[ \dot{a}_\theta^2(t) + 2\pi^2 a_\theta^2(t) \right] = \frac{\dot{a}_\theta(t)}{4} \left[ \ddot{a}_\theta(t) + 2\pi^2 a_\theta(t) \right] = 0$$

A partir de las condiciones iniciales en $t = 0$ ($a_\theta(0) = 1, \ \dot{a}_\theta(0) = \sqrt{2}\pi, \ \omega_0^2 = 2\pi^2$), el valor constante de la energía es:

$$E_0 = \frac{1}{8} \left[ 2\pi^2 + 2\pi^2 \right] = \frac{\pi^2}{2} = \frac{\omega_0^2}{4}$$

### Formulación de Residuos PINN por Modelo

Definimos el residuo temporal $r_\theta(t)$ y el residuo espaciotemporal $R_\theta(x, y, t)$:

$$r_\theta(t) = \mathcal{N}[U_\theta] - f = \ddot{a}_\theta(t) + 2\pi^2 a_\theta(t)$$

$$R_\theta(x, y, t) = \sin(\pi x)\sin(\pi y) r_\theta(t)$$

#### Modelo A
* **Conjunto de parámetros:** $\theta_A = \{A_1, A_2, \omega_1, \omega_2, b_1, b_2\}$
* **Dinámica y derivadas:**

$$a_\theta(t) = \sum_{j=1}^{2} A_j \sin(\omega_j t + b_j)$$

$$\dot{a}_\theta(t) = \sum_{j=1}^{2} A_j \omega_j \cos(\omega_j t + b_j), \quad \ddot{a}_\theta(t) = -\sum_{j=1}^{2} A_j \omega_j^2 \sin(\omega_j t + b_j)$$

* **Condiciones iniciales y residuo:**

$$a_\theta(0) = \sum_{j=1}^{2} A_j \sin(b_j), \quad \dot{a}_\theta(0) = \sum_{j=1}^{2} A_j \omega_j \cos(b_j)$$

$$r_\theta^{(A)}(t) = \ddot{a}_\theta(t) + 2\pi^2 a_\theta(t) = \sum_{j=1}^{2} A_j \left(2\pi^2 - \omega_j^2\right) \sin(\omega_j t + b_j)$$

> **Nota:** Si $\omega_k = \sqrt{2}\pi \ \forall k \in \{1, 2\}$, el Modelo A contiene exactamente la solución analítica para todo $t$, anulando idénticamente el residuo bajo esas condiciones.

#### Modelo B
* **Conjunto de parámetros:** $\theta_B = \{A, \omega_1, b_1, \omega_2, b_2\}$
* **Neurona oculta $h_1(t)$:**

$$h_1(t) = \sin(\omega_1 t + b_1)$$

$$\dot{h}_1(t) = \omega_1 \cos(\omega_1 t + b_1), \quad \ddot{h}_1(t) = -\omega_1^2 \sin(\omega_1 t + b_1) = -\omega_1^2 h_1(t)$$

* **Neurona de salida $a_\theta(t) = A \sin(\omega_2 h_1(t) + b_2)$:**

$$\begin{aligned}
\dot{a}_\theta(t) &= A \omega_2 \dot{h}_1(t) \cos(\omega_2 h_1(t) + b_2) \\
&= A \omega_1 \omega_2 \cos(\omega_1 t + b_1) \cos\left[\omega_2 \sin(\omega_1 t + b_1) + b_2\right]
\end{aligned}$$

$$\begin{aligned}
\ddot{a}_\theta(t) &= A \omega_2 \left[ \ddot{h}_1(t) \cos(\omega_2 h_1(t) + b_2) - \omega_2 \dot{h}_1^2(t) \sin(\omega_2 h_1(t) + b_2) \right] \\
&= -A \omega_1^2 \omega_2 \Big[ \sin(\omega_1 t + b_1) \cos\left[\omega_2 \sin(\omega_1 t + b_1) + b_2\right] \\
&\quad + \omega_2 \cos^2(\omega_1 t + b_1) \sin\left[\omega_2 \sin(\omega_1 t + b_1) + b_2\right] \Big]
\end{aligned}$$

* **Condiciones iniciales y residuo:**

$$a_\theta(0) = A \sin\left[\omega_2 \sin(b_1) + b_2\right]$$

$$\dot{a}_\theta(0) = A \omega_1 \omega_2 \cos(b_1) \cos\left[\omega_2 \sin(b_1) + b_2\right]$$

$$\begin{aligned}
r_\theta^{(B)}(t) &= A \left[ 2\pi^2 - \omega_1^2 \omega_2^2 \cos^2(\omega_1 t + b_1) \right] \sin\left[\omega_2 \sin(\omega_1 t + b_1) + b_2\right] \\
&\quad - A \omega_1^2 \omega_2 \sin(\omega_1 t + b_1) \cos\left[\omega_2 \sin(\omega_1 t + b_1) + b_2\right]
\end{aligned}$$

### Función de Coste PINN

Muestreando $N_{\text{ec}}$ puntos de colocación en el intervalo temporal $[0, T]$, la función de pérdida total es:

$$J(\theta) = w_{\text{ec}} J_{\text{ec}}(\theta) + w_{\text{ini}} J_{\text{ini}}(\theta)$$

donde:

$$J_{\text{ec}}(\theta) = \frac{1}{N_{\text{ec}}} \sum_{i=1}^{N_{\text{ec}}} \left| r_\theta(t_i) \right|^2 = \frac{1}{N_{\text{ec}}} \sum_{i=1}^{N_{\text{ec}}} \left| \ddot{a}_\theta(t_i) + 2\pi^2 a_\theta(t_i) \right|^2$$

$$J_{\text{ini}}(\theta) = \left| a_\theta(0) - 1 \right|^2 + \left| \dot{a}_\theta(0) - \sqrt{2}\pi \right|^2$$

* **Término adicional (regularización física para forzar invarianza energética):**

$$J_E(\theta) = \frac{1}{N_{\text{ec}}} \sum_{i=1}^{N_{\text{ec}}} \left| \frac{1}{8}\left(\dot{a}_\theta^2(t_i) + 2\pi^2 a_\theta^2(t_i)\right) - \frac{\pi^2}{2} \right|^2$$

* **Función de coste explícita para el Modelo A:**

$$\begin{aligned}
J(\theta_A) &= \frac{w_{\text{ec}}}{N_{\text{ec}}} \sum_{i=1}^{N_{\text{ec}}} \left[ \sum_{j=1}^{2} A_j \left(2\pi^2 - \omega_j^2\right) \sin(\omega_j t_i + b_j) \right]^2 \\
&\quad + w_{\text{ini}} \left[ \left( \sum_{j=1}^{2} A_j \sin(b_j) - 1 \right)^2 + \left( \sum_{j=1}^{2} A_j \omega_j \cos(b_j) - \sqrt{2}\pi \right)^2 \right]
\end{aligned}$$

* **Función de coste explícita para el Modelo B:**

$$\begin{aligned}
J(\theta_B) &= \frac{w_{\text{ec}}}{N_{\text{ec}}} \sum_{i=1}^{N_{\text{ec}}} \left[ r_\theta^{(B)}(t_i) \right]^2 \\
&\quad + w_{\text{ini}} \left[ \left( A \sin\left[\omega_2 \sin(b_1) + b_2\right] - 1 \right)^2 + \left( A \omega_1 \omega_2 \cos(b_1) \cos\left[\omega_2 \sin(b_1) + b_2\right] - \sqrt{2}\pi \right)^2 \right]
\end{aligned}$$

---

## 4. Redes de Hopfield Modernas y Mecanismo de Atención (Softmax)

Consideremos la función de energía continua para un vector de estado $q = (q_1, \dots, q_N)^\top$ y un conjunto de $p$ patrones binarios almacenados $X_{\mu j} \in \{F, T\} \equiv \{-1, +1\}$:

$$E(q) = \frac{1}{2} \sum_{j=1}^{N} q_j^2 - \frac{1}{\beta} \ln \left[ \sum_{\mu=1}^{p} \exp \left( \beta \sum_{j=1}^{N} X_{\mu j} q_j \right) \right]$$

### Derivación del Gradiente y Conexión con Softmax

Derivando el primer término respecto a la componente $q_i$:

$$\frac{\partial}{\partial q_i} \left( \frac{1}{2} \sum_{j=1}^{N} q_j^2 \right) = q_i$$

Aplicando la regla de la cadena sobre el término logarítmico:

$$\frac{\partial}{\partial q_i} \left[ -\frac{1}{\beta} \ln \left( \sum_{\mu=1}^{p} \exp \left( \beta \sum_{j=1}^{N} X_{\mu j} q_j \right) \right) \right] = -\frac{1}{\beta} \frac{\sum_{\mu=1}^{p} \exp \left( \beta \sum_{j=1}^{N} X_{\mu j} q_j \right) \cdot \frac{\partial}{\partial q_i} \left( \beta \sum_{j=1}^{N} X_{\mu j} q_j \right)}{\sum_{\nu=1}^{p} \exp \left( \beta \sum_{j=1}^{N} X_{\nu j} q_j \right)}$$

Dado que $\frac{\partial}{\partial q_i} \left( \beta \sum_{j=1}^{N} X_{\mu j} q_j \right) = \beta X_{\mu i}$, obtenemos:

$$\frac{\partial E}{\partial q_i} = q_i - \frac{\sum_{\mu=1}^{p} X_{\mu i} \exp \left( \beta \sum_{j=1}^{N} X_{\mu j} q_j \right)}{\sum_{\nu=1}^{p} \exp \left( \beta \sum_{j=1}^{N} X_{\nu j} q_j \right)} \equiv q_i - \sum_{\mu=1}^{p} w_\mu(q) X_{\mu i}$$

donde se definen los pesos normalizados tipo distribución de Boltzmann como:

$$w_\mu(q) := \frac{\exp \left( \beta \sum_{j=1}^{N} X_{\mu j} q_j \right)}{\sum_{\nu=1}^{p} \exp \left( \beta \sum_{j=1}^{N} X_{\nu j} q_j \right)} = \operatorname{softmax}(\beta X q)_\mu$$

Definiendo los vectores de patrones $X_\mu := (X_{\mu 1}, \dots, X_{\mu N})^\top$, el gradiente en notación matricial se expresa como:

$$\nabla_q E(q) = q - X^\top \operatorname{softmax}(\beta X q)$$

### Regla de Actualización por Descenso de Gradiente

Aplicando descenso de gradiente con tasa de aprendizaje $\alpha$, $q^{(t+1)} = q^{(t)} - \alpha \nabla_q E\left(q^{(t)}\right)$:

$$q_i^{(t+1)} = (1 - \alpha) q_i^{(t)} + \alpha \frac{\sum_{\mu=1}^{p} X_{\mu i} \exp \left( \beta \sum_{j=1}^{N} X_{\mu j} q_j^{(t)} \right)}{\sum_{\nu=1}^{p} \exp \left( \beta \sum_{j=1}^{N} X_{\nu j} q_j^{(t)} \right)}$$

$$q^{(t+1)} = (1 - \alpha) q^{(t)} + \alpha X^\top \operatorname{softmax}\left(\beta X q^{(t)}\right)$$

En el caso particular donde $\alpha = 1$, la actualización converge en un paso de atención directa:

$$q^{(t+1)} = X^\top \operatorname{softmax}\left(\beta X q^{(t)}\right) \implies q_i^{(t+1)} = \sum_{\mu=1}^{p} w_\mu\left(q^{(t)}\right) X_{\mu i} = \frac{\sum_{\mu=1}^{p} X_{\mu i} \exp\left(\beta S_\mu^{(t)}\right)}{\sum_{\nu=1}^{p} \exp\left(\beta S_\nu^{(t)}\right)}$$

### Análisis Físico e Informacional de los Parámetros

* **Métrica de similitud ($S_\mu^{(t)}$):** El producto interno $S_\mu^{(t)} = \sum_{j=1}^{N} X_{\mu j} q_j^{(t)} = X_\mu^\top q^{(t)}$ mide la similitud entre el estado actual de la red $q^{(t)}$ y el patrón almacenado $\mu$.
* **Frialdad informacional ($\beta$):** Actúa de forma análoga a la temperatura inversa en física estadística, $\beta = (k_B T)^{-1}$, dentro de una función de partición con distribución de Boltzmann:
  * **Régimen de frialdades bajas ($\beta \to 0, \ T \to \infty$):**
    $$\exp\left[\beta S_\mu^{(t)}\right] \approx 1 + \beta S_\mu^{(t)} \approx 1 \implies w_\mu \approx \frac{1}{p}, \quad q_i^{(t+1)} \approx \frac{1}{p} \sum_{\mu=1}^{p} X_{\mu i}$$
    Equivale a un campo lineal de una red de Hopfield clásica con pesos de Hebb. La actualización converge a un promedio aritmético global donde los patrones interfieren entre sí, impidiendo aislar una única contribución.
  * **Régimen de frialdades altas ($\beta \to \infty, \ T \to 0$):**
    Sea $\mu^*$ el índice del patrón con mayor similitud $S_{\mu^*}^{(t)}$:
    $$w_\mu\left(q^{(t)}\right) = \frac{\exp\left[\beta\left(S_\mu^{(t)} - S_{\mu^*}^{(t)}\right)\right]}{1 + \sum_{\nu \neq \mu^*} \exp\left[\beta\left(S_\nu^{(t)} - S_{\mu^*}^{(t)}\right)\right]} \xrightarrow{\beta \to \infty} \begin{cases} 1, & \mu = \mu^* \\ 0, & \mu \neq \mu^* \end{cases}$$
    En el cero absoluto informacional, la función $\operatorname{softmax}$ se comporta exactamente como un operador $\arg\max$, asignando la totalidad del peso al patrón $X_{\mu^*}$ más cercano.
  * **Frialdades negativas ($\beta < 0$):** Están prohibidas para la recuperación asociativa estándar. En mecánica cuántica y estadística corresponden a sistemas con inversión de población donde la energía está acotada entre un mínimo y un máximo, alcanzando un régimen en el que al aumentar la energía disminuye la multiplicidad de microestados.

### Criterio de Recuperación de Patrones

Tras $T$ iteraciones, el estado discreto recuperado se obtiene mediante la función signo:

$$\hat{q}_i = \operatorname{sgn}\left[q_i^{(T)}\right] = \begin{cases} +1, & q_i^{(T)} \geq 0 \\ -1, & q_i^{(T)} < 0 \end{cases}$$

Dado que el patrón objetivo cumple $X_{\mu^* i} \in \{-1, +1\}$, el producto $\hat{q}_i X_{\mu^* i}$ indica:
* $+1$: Los píxeles coinciden.
* $-1$: Los píxeles difieren.

Finalmente, el porcentaje de coincidencia se cuantifica como:

$$\text{Porcentaje de coincidencia} = \frac{100}{N} \sum_{j=1}^{N} \left[ \frac{1 + \hat{q}_j X_{\mu^* j}}{2} \right]$$
