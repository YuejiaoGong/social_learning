/*******************************************************************************/
/* This is a simple implementation of the Social Learning Algorithm (SLA).     */
/* The codes are written in C.                                                 */
/* For any questions, please contact Dr. YJ Gong (gongyuejiaoATgmail.com).         */
/* SLA_1.1,Edited on May 8th, 2016.                                           */
/*******************************************************************************/

/* add your header files here */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "function.h"

/* Change any of these parameters to match your needs */

#define POPSIZE 30				/* population size */	
#define FES 300000			    /* max. number of function evaluations */
#define TIMES 30                /* number of runs */
#define DIMS 30                 /* max. number of problem variables */

const double p_i = 0.7;         /* probability of imitation */
const double p_r = 0.2;         /* probability of randomization */
const int SN1 = 15;             /* number of model members, it is set to half of the POPSIZE by default*/
const int SN2 = POPSIZE - SN1;  /* number of non-model members */

struct Individual               /* an individual in the population */
{
	double x[DIMS];
	double fit;
};
Individual pop[POPSIZE];       /* population */
Individual newpop[POPSIZE];    /* new population, replaces the old population */

double lbound, ubound;         /* the lower and upper bounds of variables */


/* control parameters used by the program */

int    fes;                    /* current number of function evaluations */
double gbestval;               /* the best fitness value found so far */
int    gbestind;               /* index of the best individual with gbestval */
double t_Val[DIMS];            /* t-value on each dimension by t-test */
double AT, mAT;                /* attention threshold */
						       /* AT = the absolute value of the t-value on a random dimension */
						       /* mAT = -AT */

/* declaration of functions used by this social learning algorithm */
void Initialize();
void Evaluate();
void Attention();
void Reproduction_and_Reinforcement();
void Motivation();
void Process();

double (*function_name)(double pos[],int dim);
double randval (double low,double high);
int    cmp (const void *a , const void *b);
double t_test (double sample1[], int SIZE1, double sample2[], int SIZE2);



/***************************************************************/
/* Initialization function: Initializes the values of          */
/* individuals within the variables bounds [lbound, ubound].   */
/***************************************************************/

void Initialize()
{
	int i, j;

	fes = 0;
	for(i = 0; i < POPSIZE; i++){
		for(j = 0; j < DIMS; j++){
			pop[i].x[j] = randval(lbound,ubound);			
		}	
	}
	gbestval = 1e300;   /* set the gbestval to a worst value you can have. */
	                    /* if you are dealing with a maximization problem, */
	                    /* -1e300 should be used here                      */
}

/*************************************************************/
/* Fitness evaluation: this takes a user defined function.   */
/* "function_name" indicates the object function you are     */
/* testing, which should be initialized before used          */
/*************************************************************/

void Evaluate()
{
	int i;	

	for(i = 0; i < POPSIZE; i++){
		fes++;
		pop[i].fit = function_name(pop[i].x,DIMS);
		if(pop[i].fit - gbestval < 0 ){
			gbestind = i;
			gbestval = pop[i].fit;
		}
	}	
}

/*************************************************************/
/* Attention operator: sort the population by fitness,       */
/* divide the population into s1[] and s2[], use student's   */
/* t-test to compare each dimension of s1 and s2,            */
/* record the t-values on each dimension in t_Val[].         */
/*************************************************************/

void Attention()
{                       
	int i, i1, j;
	double s1[SN1], s2[SN2];

	qsort(pop, POPSIZE, sizeof(pop[0]),cmp);      /* sort the population from best to worst */
	for(j = 0; j < DIMS; j++){                    /* copy the sorted population to s1[] and s2[] */
		for(i=0; i<SN1; i++)
			s1[i]=pop[i].x[j];
		i1=SN1;
		for(i=0; i<SN2; i++){
			s2[i]=pop[i1++].x[j];
		}
		t_Val[j] = t_test(s1, SN1, s2, SN2);	  /* perform students' t-test */
	}
	AT = fabs(t_Val[rand()%DIMS]);                /* calculate the attention threshold AT and -AT */
	mAT = 0-AT;                                 
}

/*************************************************************/
/* Reproduction_and_Reinforcement operators: this function   */
/* combines the reproduction and reinforcement operators in  */
/* SLA. For each dimension, if the t-value locates beyond    */
/* [mAT, AT], the individual copies the value of a random    */
/* model and then undergoes positive or negative             */
/* reinforcement; otherwise the individual explores the      */
/* dimensions by probability p_i and p_r.                    */
/*************************************************************/

void Reproduction_and_Reinforcement()
{
	int i, j, r, r1;
	double nd, delta;

	for(i = 0; i < POPSIZE; i++){                           /* update each individual */
		for(j = 0; j < DIMS; j++){
			do{	r = rand()%SN1; } while(r == i);            /* r is a random model member */
			do{ r1 = rand()%POPSIZE; } while(r1 == i);	    /* r1 is a random individual selected */
														    /* from the entire population         */
			nd = randval(0, 1);
			delta = fabs(pop[r].x[j]-pop[i].x[j]);          /* reinforcement step length */

			if(t_Val[j] >= AT){
				newpop[i].x[j] = pop[r].x[j] + nd * delta;  /* imitation of good members and positive reinforcement */
			}
			else if(t_Val[j] <= mAT){
				newpop[i].x[j] = pop[r].x[j] - nd * delta;  /* imitation of good members and negative reinforcement */
			}
			else{
				/* explores the dimension */
				if(randval(0,1) < p_i){
					newpop[i].x[j] = pop[r1].x[j];          /* random imitation */
				}
				else if(randval(0,1) < p_r){
					newpop[i].x[j]=randval(lbound,ubound);  /* reinitialization */
				}
				else	
					newpop[i].x[j] = pop[i].x[j];             
			}

			/* boundary control, comment this if the problem does not have a boundary */
			if(newpop[i].x[j] > ubound)
				newpop[i].x[j] = ubound - 0.5 * (newpop[i].x[j]-ubound);
			else if(newpop[i].x[j]<lbound)
				newpop[i].x[j] = lbound + 0.5 * (lbound-newpop[i].x[j]);		
		}
	}				
}

/*************************************************************/
/* Motivation operator: new individuals replace the old      */
/* ones if they have better fitness values.                  */
/*************************************************************/

void Motivation()
{
	int i;	

	for(i = 0; i < POPSIZE; i++){
		fes++;
		newpop[i].fit = function_name(newpop[i].x,DIMS);
		if(newpop[i].fit - pop[i].fit <= 0){
			pop[i] = newpop[i];
			if(pop[i].fit - gbestval <= 0){
				gbestind = i;
				gbestval = pop[i].fit;
			}
		}
	}	
}

/*************************************************************/
/* Process: this function contains the overall procedure for */
/* running the social learning algorithm.                    */
/*************************************************************/

void Process()
{
	Initialize();
	Evaluate();
	while(fes < FES){	    /* loop until terminal condition */
		Attention();
		Reproduction_and_Reinforcement();
		Motivation();
	}
}

/*************************************************************/
/* The main function                                         */
/*************************************************************/

void main()
{
	int i;

	function_name = f1;            /* set the objective function here */
	lbound = -100; ubound = 100;   /* set the variable range */
	for(i = 0; i < TIMES; i++){
		Process();                 /* run the SLA algorithm */

        /* you can output results using gbestval and pop[gbestind] here, e.g., */
		printf("%g\n", gbestval);
	}
}	




/* auxiliary functions */

/***********************************************************/
/* Random value generator: generates a value within bounds */
/***********************************************************/

double randval (double low,double high) 
{
	return (double(rand())/RAND_MAX) * (high-low) + low;
}

/*************************************************************/
/* Comparison: compares the fitness of two individuals,      */
/* a and b, in the population.                               */
/*************************************************************/

int cmp (const void *a , const void *b) 
{ 
	struct Individual * p = (struct Individual *) a;
	struct Individual * q = (struct Individual *) b;
	return (p->fit - q->fit >0) ? 1 : -1; ; 
} 

/*************************************************************/
/* Student's t_test: conduct student's t-test to compare the */
/* values of sample1[] and sample2[] with SIZE1 and SIZE2.   */
/* return the t-value.                                       */
/*************************************************************/

double t_test(double sample1[], int SIZE1, double sample2[], int SIZE2)
{
	int i;
	double t_value;
	double mean_1 = 0, mean_2 = 0, SS_1 = 0,SS_2 = 0, SS_1_2, S_1_2;
	int df = SIZE1 - 1 + SIZE2 - 1;            /* degree of freedom */

	/* calculate mean */
	for(i=0; i<SIZE1; i++)  mean_1+=sample1[i];
	for(i=0; i<SIZE2; i++)	mean_2+=sample2[i];
	mean_1/=SIZE1;
	mean_2/=SIZE2;

	/* calculate SS */
	for(i=0; i<SIZE1; i++)	SS_1+=(sample1[i]-mean_1)*(sample1[i]-mean_1);
	for(i=0; i<SIZE2; i++)	SS_2+=(sample2[i]-mean_2)*(sample2[i]-mean_2);	

	/* calculate S_1_2 */
	SS_1_2=((SS_1+SS_2)/(df))*(1.0/SIZE1+1.0/SIZE2);
	S_1_2=sqrt(SS_1_2);

	/* calculate and return t-value */
	if(S_1_2 == 0) t_value = 0;
	else
		t_value = (mean_1-mean_2) / S_1_2;	
	return t_value;
}