#include"Util.h"

Util* Util::util = 0;


Util::Util()
{
	
}


void Util::parse_config_file(CONFIG con)
{
	if (con == CONFIG::BASE_IMAGE_PATH)
	{
		IMAGE_BASE_PATH_KEY = "BASE_IMAGE_PATH";
		IMAGE_BASE_PATH_VALUE = find_config_value_by_key(IMAGE_BASE_PATH_KEY);
		setImage_Base_Path(IMAGE_BASE_PATH_VALUE);
	}
}

void Util::setImage_Base_Path(string value)
{
	IMAGE_BASE_PATH_VALUE = value;
}

string Util::getImage_Base_Path()
{
	return IMAGE_BASE_PATH_VALUE;
}


string Util::find_config_value_by_key(string key)
{
	auto config_it = find_if(config_list.begin(), config_list.end(), [key](Config con)->bool{return con.config_key == key; });
	return config_it->config_value;
}

vector<string> Util::split(string data , string delim)
{
	vector<string>split_list;
	/*size_t pos = 0;
	std::string token;
	while ((pos = data.find(delim)) != std::string::npos) {
		token = data.substr(0, pos);
		//std::cout << token << std::endl;
		split_list.push_back(token);
		//data.erase(0, pos + delim.length());
	}*/
	size_t len = data.length() + 1;
	char* str = new char[len];
	strcpy(str, data.c_str());
	char * pch;
	//printf("Splitting string \"%s\" into tokens:\n", str);
	len = delim.length() + 1;
	char* c_delim = new char[len];
	strcpy(c_delim, delim.c_str());
	pch = strtok(str, c_delim);
	while (pch != NULL)
	{
		string token(pch);
		split_list.push_back(token);
		pch = strtok(NULL, c_delim);
	}




	return split_list;
}


void Util::read_config_file(string config_file)
{
	string line;
	ifstream config_reader(config_file , ios::in);
	if (config_reader.is_open())
	{
		while (getline(config_reader, line))
		{
			vector<string>list = split(line, ",");
			Config con;
			con.config_key = list[0];
			con.config_value = list[1];
			config_list.push_back(con);
		}
	}
	else
	{
		cout << "Unable to open file" << endl;
	}
}


Util* Util::getInstance()
{
	if (!util)
	{
		util = new Util();
		return util;
	}
	else
	{
		return util;
	}
}