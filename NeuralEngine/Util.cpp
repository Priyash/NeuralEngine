#include"Util.h"

Util* Util::util = 0;


Util::Util()
{
	
}


void Util::parse_config_file(CONFIG_ID con)
{
	if (con == CONFIG_ID::BASE_IMAGE_PATH)
	{
		string IMAGE_BASE_PATH_VALUE = find_config_value_by_key(getKey(con));
		setImage_Base_Path(IMAGE_BASE_PATH_VALUE);
	}
	else if (con == CONFIG_ID::IMAGE_RESIZE)
	{
		string IMAGE_RESIZE = getKey(con);
		vector<string>resize_list = split(IMAGE_RESIZE, ",");

		IMAGE_RESIZE_WIDTH_VALUE = find_config_value_by_key(resize_list[0]);

		IMAGE_RESIZE_HEIGHT_VALUE = find_config_value_by_key(resize_list[1]);

		setResizeValue(stoi(IMAGE_RESIZE_WIDTH_VALUE), stoi(IMAGE_RESIZE_HEIGHT_VALUE));
		resize_list.clear();
	}
	else if (con == CONFIG_ID::IMAGE_OUTPUT_PATH)
	{
		string Image_out_path = find_config_value_by_key(getKey(con));
		setImageOutputPath(Image_out_path);
	}
	else if (con == CONFIG_ID::IMAGE_BATCH_SIZE)
	{
		string Image_Batch_Size = find_config_value_by_key(getKey(con));
		setImageBatchSize(stoi(Image_Batch_Size));
	}

	else if (con == CONFIG_ID::IMAGE_INPUT_FEATURE_MAP)
	{
		string Image_Channel = find_config_value_by_key(getKey(con));
		setImageInputFeatureMap(stoi(Image_Channel));
	}
}


string Util::getKey(CONFIG_ID con)
{
	switch (con)
	{
	case CONFIG_ID::BASE_IMAGE_PATH:
		return "BASE_IMAGE_PATH";
		break;

	case CONFIG_ID::IMAGE_RESIZE:

		return "IMAGE_RESIZE_WIDTH,IMAGE_RESIZE_HEIGHT";
		break;
	}
}


void Util::setImage_Base_Path(string value)
{
	IMAGE_BASE_PATH_VALUE = value;
}
void Util::setImageOutputPath(string value)
{
	IMAGE_OUTPUT_PATH_VALUE = value;
}

string Util::getImageOutputPath()
{
	return IMAGE_OUTPUT_PATH_VALUE;
}

string Util::getImage_Base_Path()
{
	return IMAGE_BASE_PATH_VALUE;
}


void Util::setResizeValue(int width, int height)
{
	size.width = width;
	size.height = height;
}

Resize Util::getResizeValue()
{
	return size;
}



void Util::setImageBatchSize(int value)
{
	this->IMAGE_BATCH_SIZE_VALUE = value;
}

int Util::getImageBatchSize()
{
	return IMAGE_BATCH_SIZE_VALUE;
}

void Util::setImageInputFeatureMap(int value)
{
	this->IMAGE_INPUT_FEATURE_MAP_VALUE = value;
}

int Util::getImageInputFeatureMap()
{
	return IMAGE_INPUT_FEATURE_MAP_VALUE;
}












string Util::find_config_value_by_key(string key)
{
	auto config_it = find_if(config_list.begin(), config_list.end(), [key](Config con)->bool{return con.config_key == key; });
	return config_it->config_value;
}

vector<string> Util::split(string data , string delim)
{
	vector<string>split_list;
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