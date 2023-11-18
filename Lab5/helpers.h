

struct variable_node
{
    const char* name;
    double value;
    struct variable_node* next;
};

typedef struct variable_node variable_node;

variable_node* get_variable(const char* name);

void add_variable(const char* name, double value);

void init_variable_table();

int process_unary_function(const char* func, double val, double* res);