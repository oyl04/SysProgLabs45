#define NODES_SIZE 10 

typedef struct node
{
    const char* name;
    double value;
    struct node* next_elements[NODES_SIZE];
} node;

node* create_node(const char* name);
node* create_node_value(const char* name, double value);
node* create_node_list_2(const char* name, node* one, node* two);
node* create_node_list_3(const char* name, node* one, node* two, node* three);

void add_sub_node(node* main_node, node* sub_node);

void print_syntax_tree(node* node);