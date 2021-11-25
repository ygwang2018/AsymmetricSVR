# AsymmetricSVR_Ryan
The source code for the case study of Chapter 6 of Ryan's PhD thesis. In this chapter, an asymmetric SVR framework is proposed for minimizing the economical costs for load forecasting.

Support vector regression with asymmetric loss for optimal electric load forecasting

In energy demand forecasting, the objective function is often symmetric, implying that over-prediction errors and under-prediction errors have the same consequences. In practice, these two types of errors generally incur very different costs. To accommodate this, we propose a machine learning algorithm with a cost-oriented asymmetric loss function in the training procedure. Specifically, we develop a new support vector regression incorporating a linear-linear cost function and the insensitivity parameter for sufficient fitting. The electric load data from the state of New South Wales in Australia is used to show the superiority of our proposed framework. Compared with the basic support vector regression, our new asymmetric support vector regression framework for multi-step load forecasting results in a daily economic cost reduction ranging from 42.19%  to 57.39%, depending on the actual cost ratio of the two types of errors.

See: https://www.sciencedirect.com/science/article/pii/S0360544221002188
