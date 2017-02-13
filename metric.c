#include <python3.5m/Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

float * getArray(PyObject * obj) {
    unsigned int len = PyObject_Length(obj);
    PyObject * iter = PyObject_GetIter(obj);
    float * res;
    res = (float *) malloc(len * sizeof(float));
    if (res == NULL) {
        printf("res is null\n");
        exit(1);
    }
    if (!iter) {
        return NULL;
    }
    PyObject *next;
    for (int i = 0; i < len; ++i) {
        next = PyIter_Next(iter);
        if (!next) {
            break;
        }

        if (!PyFloat_Check(next)) {
            // error, we were expecting a floating point value
        }
        res[i] = PyFloat_AsDouble(next);
        /*printf("parsed %i elem \n", i);*/
    }
    return res;
}

static PyObject *
metric_sum(PyObject *self, PyObject *args)
{
    /*float sum;*/
    /*PyObject *aObj;*/
    /*PyObject *bObj;*/
    /*unsigned int len;*/
    /*double foo = 0;*/

    /*if (!PyArg_ParseTuple(args, "OO", &aObj, &bObj))*/
        /*return NULL;*/

    /*float * a_vector = getArray(aObj);*/
    /*float * b_vector = getArray(bObj);*/

    /*for (unsigned int i = 0; i < len; ++i) {*/
        /*foo += ((*(a_vector + i)) + (*(b_vector + i)));*/
        /*[>printf("summed %i pair \n", i);<]*/
    /*}*/

    /*free(a_vector);*/
    /*free(b_vector);*/
    return PyFloat_FromDouble(1.);
}

static PyObject *
metric_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyMethodDef MetricMethods[] = {
    {"sum",  metric_sum, METH_VARARGS,
     "Execute sum."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef metricmodule = {
   PyModuleDef_HEAD_INIT,
   "metric",   // name of module
   "Test metric module documentation", // module documentation, may be NULL
   -1,       // size of per-interpreter state of the module,
                // or -1 if the module keeps state in global variables.
   MetricMethods
};

static PyObject *MetricError;
PyMODINIT_FUNC
PyInit_metric(void)
{
    PyObject *m;

    m = PyModule_Create(&metricmodule);
    if (m == NULL)
        return NULL;

    MetricError = PyErr_NewException("metric.error", NULL, NULL);
    Py_INCREF(MetricError);
    PyModule_AddObject(m, "error", MetricError);
    return m;
}
